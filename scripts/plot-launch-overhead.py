#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Plot launch-overhead results from bench-launch-overhead.py."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def _maybe_set_backend() -> None:
    if os.environ.get("DISPLAY") is None and os.name != "nt":
        import matplotlib

        matplotlib.use("Agg")


def _load_input(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return _normalize_json(data)
    except json.JSONDecodeError:
        return _parse_table(text)


def _normalize_json(data: object) -> list[dict]:
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON list of results or a single result dict."
        )
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each JSON entry must be a dict.")
        if "label" not in entry or "results" not in entry:
            raise ValueError(
                "Each JSON entry must include 'label' and 'results'."
            )
    return data


def _split_cols(line: str) -> list[str]:
    return [col for col in re.split(r"\s{2,}|\t+", line.strip()) if col]


def _parse_table(text: str) -> list[dict]:
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Launch overhead (us/launch):"):
            start = idx
            break
    if start is None or start + 2 >= len(lines):
        raise ValueError("Could not locate launch-overhead table in text.")

    header = _split_cols(lines[start + 1])
    if not header or header[0] != "args":
        raise ValueError("Malformed table header.")
    labels = header[1:]
    results: dict[str, dict[str, dict[str, float]]] = {
        label: {} for label in labels
    }

    row_idx = start + 2
    while row_idx < len(lines):
        line = lines[row_idx].rstrip()
        if not line.strip() or line.strip().startswith("Deltas vs baseline"):
            break
        cols = _split_cols(line)
        if len(cols) < 2:
            row_idx += 1
            continue
        arg = cols[0]
        for label, cell in zip(labels, cols[1:]):
            match = re.search(
                r"([+-]?\d+(?:\.\d+)?)\s*\+/-\s*([+-]?\d+(?:\.\d+)?)",
                cell,
            )
            if not match:
                raise ValueError(f"Could not parse cell '{cell}'.")
            mean = float(match.group(1))
            stdev = float(match.group(2))
            results[label][arg] = {"mean_us": mean, "stdev_us": stdev}
        row_idx += 1

    return [{"label": label, "results": results[label]} for label in labels]


def _extract_series(
    results: list[dict],
) -> tuple[list[int], list[str], dict, dict]:
    labels = [entry["label"] for entry in results]
    arg_keys = sorted(
        {int(k) for entry in results for k in entry["results"].keys()}
    )
    means: dict[str, list[float]] = {}
    stdevs: dict[str, list[float]] = {}
    for entry in results:
        label = entry["label"]
        means[label] = []
        stdevs[label] = []
        for arg in arg_keys:
            record = entry["results"].get(str(arg))
            if not record:
                raise ValueError(f"Missing arg {arg} for label '{label}'.")
            means[label].append(float(record.get("mean_us", 0.0)))
            stdevs[label].append(float(record.get("stdev_us", 0.0)))
    return arg_keys, labels, means, stdevs


def _select_baseline(labels: list[str], baseline: str | None) -> str:
    if baseline:
        if baseline not in labels:
            raise ValueError(
                f"Baseline '{baseline}' not found in labels: {labels}"
            )
        return baseline
    return labels[0]


def _summarize_device(results: list[dict]) -> str | None:
    devices = []
    runtimes = []
    for entry in results:
        device = entry.get("device")
        runtime = entry.get("cuda_runtime_version")
        if device:
            name = device.get("name")
            cc = device.get("cc")
            if name and cc:
                devices.append((name, tuple(cc)))
        if runtime:
            runtimes.append(tuple(runtime))
    device_line = None
    if devices and len(set(devices)) == 1:
        name, cc = devices[0]
        device_line = f"{name} (CC {cc[0]}.{cc[1]})"
    runtime_line = None
    if runtimes and len(set(runtimes)) == 1:
        runtime_line = f"CUDA runtime {runtimes[0][0]}.{runtimes[0][1]}"
    if device_line and runtime_line:
        return f"{device_line} • {runtime_line}"
    if device_line:
        return device_line
    if runtime_line:
        return runtime_line
    return None


def _sanitize_svg(path: Path) -> None:
    if path.suffix.lower() != ".svg":
        return
    text = path.read_text(encoding="utf-8")
    sanitized = "\n".join(line.rstrip() for line in text.splitlines())
    if text.endswith("\n"):
        sanitized += "\n"
    if sanitized != text:
        path.write_text(sanitized, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot launch-overhead results from bench-launch-overhead.py."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to results JSON (preferred) or bench stdout text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (png/svg/pdf). If omitted, show the plot.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Label to use as baseline (defaults to first entry).",
    )
    parser.add_argument(
        "--delta",
        choices=("pct", "us", "none"),
        default="pct",
        help="Show delta vs baseline as percent, microseconds, or not at all.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title override.",
    )
    parser.add_argument(
        "--no-sns",
        action="store_true",
        help="Disable seaborn styling even if installed.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI when saving images.",
    )
    args = parser.parse_args()

    _maybe_set_backend()

    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None

    results = _load_input(args.input)
    arg_keys, labels, means, stdevs = _extract_series(results)
    baseline_label = _select_baseline(labels, args.baseline)

    if sns is not None and not args.no_sns:
        sns.set_theme(style="whitegrid")
        palette = sns.color_palette(n_colors=len(labels))
    else:
        palette = plt.cm.tab10.colors

    nrows = 2 if args.delta != "none" else 1
    height = 6.5 if nrows == 2 else 4.0
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(8.5, height),
        sharex=True,
        constrained_layout=True,
    )
    if nrows == 1:
        ax_abs = axes
        ax_delta = None
    else:
        ax_abs, ax_delta = axes

    x = np.array(arg_keys)
    for idx, label in enumerate(labels):
        ax_abs.errorbar(
            x,
            means[label],
            yerr=stdevs[label],
            label=label,
            color=palette[idx % len(palette)],
            marker="o",
            linewidth=2,
            capsize=3,
        )
    ax_abs.set_ylabel("Launch overhead (us/launch)")
    ax_abs.set_xticks(x)
    ax_abs.grid(True, axis="y", alpha=0.3)
    ax_abs.legend(frameon=False, title="Repo", ncol=min(3, len(labels)))

    if ax_delta is not None:
        base = np.array(means[baseline_label])
        ax_delta.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
        for idx, label in enumerate(labels):
            if label == baseline_label:
                continue
            delta = np.array(means[label]) - base
            if args.delta == "pct":
                delta = np.where(base != 0, (delta / base) * 100, 0.0)
                ylabel = "Delta vs baseline (%)"
            else:
                ylabel = "Delta vs baseline (us/launch)"
            ax_delta.plot(
                x,
                delta,
                label=label,
                color=palette[idx % len(palette)],
                marker="o",
                linewidth=2,
            )
        ax_delta.set_ylabel(ylabel)
        ax_delta.set_xlabel("Kernel args")
        ax_delta.grid(True, axis="y", alpha=0.3)
        ax_delta.legend(frameon=False, title="Repo", ncol=min(3, len(labels)))
    else:
        ax_abs.set_xlabel("Kernel args")

    title = args.title or "CUDA kernel launch overhead"
    subtitle = _summarize_device(results)
    if subtitle:
        title = f"{title}\n{subtitle}"
    fig.suptitle(title, fontsize=12)

    if args.output:
        fig.savefig(args.output, dpi=args.dpi)
        _sanitize_svg(args.output)
        print(f"Wrote {args.output}")
        return 0

    try:
        plt.show()
    except Exception:
        fallback = Path("launch-overhead.png")
        fig.savefig(fallback, dpi=args.dpi)
        print(f"Wrote {fallback}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
