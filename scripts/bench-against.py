# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import argparse
import contextlib
import os
import subprocess
import tempfile


@contextlib.contextmanager
def chdir(path):
    """Change directory context manager."""
    curdir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(curdir)


@contextlib.contextmanager
def git_worktree():
    """Create a temporary git worktree that can be reused for multiple commits."""
    # Get the git root directory
    git_dir = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.rstrip()

    # Create a temporary directory for the worktree
    with tempfile.TemporaryDirectory(prefix="bench-against-") as worktree_path:
        # Create the worktree starting from the current HEAD
        # We'll checkout different commits within this worktree
        # Do this before the try block so cleanup only happens if creation succeeds
        subprocess.run(
            ["git", "worktree", "add", "--detach", worktree_path, "HEAD"],
            check=True,
            cwd=git_dir,
        )

        try:
            with chdir(worktree_path):
                yield
        finally:
            # Remove the worktree
            with contextlib.suppress(subprocess.CalledProcessError):
                subprocess.run(
                    ["git", "worktree", "remove", "--force", worktree_path],
                    check=True,
                    cwd=git_dir,
                )


def main(args):
    baseline = subprocess.check_output(
        ["git", "rev-parse", args.baseline], text=True
    ).strip()
    proposed = subprocess.check_output(
        ["git", "rev-parse", args.proposed], text=True
    ).strip()
    with git_worktree():
        # Checkout baseline and run benchmarks
        subprocess.run(["git", "checkout", baseline], check=True)
        subprocess.run(
            ["pixi", "reinstall", "-q", "-e", args.env, "numba-cuda"],
            check=True,
        )
        subprocess.run(
            ["pixi", "run", "-q", "-e", args.env, "bench"], check=True
        )

        # Checkout proposed and run comparison
        subprocess.run(["git", "checkout", proposed], check=True)
        subprocess.run(
            ["pixi", "reinstall", "-q", "-e", args.env, "numba-cuda"],
            check=True,
        )
        subprocess.run(
            ["pixi", "run", "-q", "-e", args.env, "benchcmp"], check=True
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare two git refs' benchmarks.")

    p.add_argument(
        "baseline", help="Git ref of baseline to compare against. E.g.: HEAD~"
    )
    p.add_argument(
        "proposed", help="Git ref to compare against the baseline. E.g.: HEAD"
    )
    p.add_argument(
        "-e",
        "--env",
        default="cu-12-9-py312",
        help="Environment to run benchmarks in.",
    )

    main(p.parse_args())
