#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import argparse
import json
import os
import statistics
import subprocess
import sys
import time


DEFAULT_LOOPS = {
    0: 100_000,
    1: 100_000,
    2: 100_000,
    3: 100_000,
    4: 10_000,
}
DEFAULT_REPEATS = 7


def _parse_repo(spec):
    if "=" not in spec:
        raise ValueError("Repo spec must be in the form label=/path/to/repo.")
    label, path = spec.split("=", 1)
    label = label.strip()
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if not label:
        raise ValueError("Repo spec label cannot be empty.")
    return label, path


def _parse_loops(spec):
    if spec is None:
        return DEFAULT_LOOPS
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 5:
        raise ValueError("Loops must be a comma-separated list of 5 integers.")
    loops = {}
    for idx, value in enumerate(parts):
        loops[idx] = int(value)
    return loops


def _git_rev(path):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
    ).strip()


def _pip_install(repo_path, python):
    subprocess.run(
        [
            python,
            "-m",
            "pip",
            "install",
            "-e",
            repo_path,
            "--no-deps",
        ],
        check=True,
    )


def _run_worker(label, loops, repeats, json_only):
    import numpy as np
    import numba
    from numba import cuda
    from numba.cuda.core import config

    if config.ENABLE_CUDASIM:
        raise RuntimeError("CUDA simulator enabled; benchmarks require GPU.")

    cuda.current_context()

    arrs = [cuda.device_array(10_000, dtype=np.float32) for _ in range(4)]

    @cuda.jit("void()")
    def some_kernel_1():
        return

    @cuda.jit("void(float32[:])")
    def some_kernel_2(arr1):
        return

    @cuda.jit("void(float32[:],float32[:])")
    def some_kernel_3(arr1, arr2):
        return

    @cuda.jit("void(float32[:],float32[:],float32[:])")
    def some_kernel_4(arr1, arr2, arr3):
        return

    @cuda.jit("void(float32[:],float32[:],float32[:],float32[:])")
    def some_kernel_5(arr1, arr2, arr3, arr4):
        return

    kernels = [
        ("0", some_kernel_1, ()),
        ("1", some_kernel_2, (arrs[0],)),
        ("2", some_kernel_3, (arrs[0], arrs[1])),
        ("3", some_kernel_4, (arrs[0], arrs[1], arrs[2])),
        ("4", some_kernel_5, (arrs[0], arrs[1], arrs[2], arrs[3])),
    ]

    results = {}
    for idx, (name, kernel, args) in enumerate(kernels):
        loop_count = loops[idx]
        kernel[1, 1](*args)
        cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(loop_count):
                kernel[1, 1](*args)
            cuda.synchronize()
            elapsed = time.perf_counter() - start
            samples.append(elapsed / loop_count)
        mean_s = statistics.mean(samples)
        stdev_s = statistics.stdev(samples) if repeats > 1 else 0.0
        results[name] = {
            "loops": loop_count,
            "mean_us": mean_s * 1e6,
            "stdev_us": stdev_s * 1e6,
        }

    device = cuda.get_current_device()
    payload = {
        "label": label,
        "numba_version": numba.__version__,
        "device": {
            "name": device.name.decode()
            if hasattr(device.name, "decode")
            else device.name,
            "cc": device.compute_capability,
        },
        "cuda_runtime_version": cuda.runtime.get_version(),
        "results": results,
        "repeats": repeats,
    }

    if not json_only:
        print(
            f"{label}: {payload['numba_version']} "
            f"CUDA {payload['cuda_runtime_version']}"
        )
    print(json.dumps(payload))


def _format_us(value):
    return f"{value:.2f}"


def _print_table(results):
    labels = [r["label"] for r in results]
    baseline = results[0]["label"]
    print("Launch overhead (us/launch):")
    header = ["args"] + labels
    rows = []
    for arg in ["0", "1", "2", "3", "4"]:
        row = [arg]
        for r in results:
            mean = r["results"][arg]["mean_us"]
            stdev = r["results"][arg]["stdev_us"]
            row.append(f"{_format_us(mean)} +/- {_format_us(stdev)}")
        rows.append(row)

    col_widths = [
        max(len(row[i]) for row in [header] + rows) for i in range(len(header))
    ]
    fmt = "  ".join(f"{{:<{width}}}" for width in col_widths)
    print(fmt.format(*header))
    for row in rows:
        print(fmt.format(*row))

    if len(results) > 1:
        print("")
        print(f"Deltas vs {baseline}:")
        header = ["args"] + labels[1:]
        delta_rows = []
        for arg in ["0", "1", "2", "3", "4"]:
            base_mean = results[0]["results"][arg]["mean_us"]
            row = [arg]
            for r in results[1:]:
                mean = r["results"][arg]["mean_us"]
                delta = mean - base_mean
                pct = (delta / base_mean) * 100 if base_mean else 0.0
                row.append(f"{_format_us(delta)} ({pct:+.1f}%)")
            delta_rows.append(row)

        col_widths = [
            max(len(row[i]) for row in [header] + delta_rows)
            for i in range(len(header))
        ]
        fmt = "  ".join(f"{{:<{width}}}" for width in col_widths)
        print(fmt.format(*header))
        for row in delta_rows:
            print(fmt.format(*row))


def _run_driver(args):
    repos = [_parse_repo(spec) for spec in args.repo]
    loops = _parse_loops(args.loops)
    results = []
    for label, path in repos:
        sha = _git_rev(path)
        if not args.no_install:
            _pip_install(path, args.python)
        output = subprocess.check_output(
            [
                args.python,
                os.path.abspath(__file__),
                "--run",
                "--label",
                label,
                "--loops",
                args.loops
                if args.loops
                else ",".join(str(loops[i]) for i in range(5)),
                "--repeats",
                str(args.repeats),
            ],
            text=True,
        )
        last_line = output.strip().splitlines()[-1]
        payload = json.loads(last_line)
        payload["repo"] = path
        payload["sha"] = sha
        results.append(payload)

    _print_table(results)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
        print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description=("Benchmark kernel launch overhead across repos.")
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="Repo spec as label=/path (repeatable).",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip pip install -e for repos.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use.",
    )
    parser.add_argument(
        "--loops",
        default=None,
        help="Comma-separated loops for 0..4 args.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of repeats for each kernel.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run in worker mode (internal).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Label for worker mode.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Suppress non-JSON output in worker mode.",
    )
    args = parser.parse_args()

    if args.run:
        loops = _parse_loops(args.loops)
        _run_worker(args.label, loops, args.repeats, args.json_only)
        return

    if not args.repo:
        parser.error("--repo must be provided at least once.")

    _run_driver(args)


if __name__ == "__main__":
    main()
