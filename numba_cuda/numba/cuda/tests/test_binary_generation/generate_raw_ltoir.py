# Copyright (c) 2024, NVIDIA CORPORATION.

import argparse
import pathlib
import subprocess
import sys

from cuda import nvrtc

# Magic number found at the start of an LTO-IR file
LTOIR_MAGIC = 0x7F4E43ED


def check(args):
    """
    Abort and print an error message in the presence of an error result.

    Otherwise:
    - Return None if there were no more arguments,
    - Return the singular argument if there was only one further argument,
    - Return the tuple of arguments if multiple followed.
    """

    result, *args = args
    value = result.value

    if value:
        error_string = check(nvrtc.nvrtcGetErrorString(result)).decode()
        msg = f"NVRTC error, code {value}: {error_string}"
        print(msg, file=sys.stderr)
        sys.exit(1)

    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return args


def determine_include_flags():
    # Inspired by the logic in FindCUDAToolkit.cmake. We need the CUDA include
    # paths because NVRTC doesn't add them by default, and we can compile a
    # much broader set of test files if the CUDA includes are available.

    # We invoke NVCC in verbose mode ("-v") and give a dummy filename, without
    # which it won't produce output.

    cmd = ["nvcc", "-v", "__dummy"]
    cp = subprocess.run(cmd, capture_output=True)

    # Since the dummy file doesn't actually exist, NVCC is expected to exit
    # with an error code of 1.
    rc = cp.returncode
    if rc != 1:
        print(f"Unexpected return code ({rc}) from `nvcc -v`. Expected 1.")
        return None

    output = cp.stderr.decode()
    lines = output.splitlines()

    includes_lines = [line for line in lines if line.startswith("#$ INCLUDES=")]
    if len(includes_lines) != 1:
        print(f"Expected exactly one INCLUDES line. Got {len(includes_lines)}.")
        return None

    # Parse out the arguments following "INCLUDES=" - these are a space
    # separated list of strings that are potentially quoted.

    quoted_flags = includes_lines[0].split("INCLUDES=")[1].strip().split()
    include_flags = [flag.strip('"') for flag in quoted_flags]
    print(f"Using CUDA include flags: {include_flags}")

    return include_flags


def get_ltoir(source, name, arch):
    """Given a CUDA C/C++ source, compile it and return the LTO-IR."""

    program = check(
        nvrtc.nvrtcCreateProgram(source.encode(), name.encode(), 0, [], [])
    )

    cuda_include_flags = determine_include_flags()
    if cuda_include_flags is None:
        print("Error determining CUDA include flags. Exiting.", file=sys.stderr)
        sys.exit(1)

    options = [
        f"--gpu-architecture={arch}",
        "-dlto",
        "-rdc",
        "true",
        *cuda_include_flags,
    ]
    options = [o.encode() for o in options]

    result = nvrtc.nvrtcCompileProgram(program, len(options), options)

    # Report compilation errors back to the user
    if result[0] == nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION:
        log_size = check(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * log_size
        check(nvrtc.nvrtcGetProgramLog(program, log))
        print("NVRTC compilation error:\n", file=sys.stderr)
        print(log.decode(), file=sys.stderr)
        sys.exit(1)

    # Handle other errors in the standard way
    check(result)

    ltoir_size = check(nvrtc.nvrtcGetLTOIRSize(program))
    ltoir = b" " * ltoir_size
    check(nvrtc.nvrtcGetLTOIR(program, ltoir))

    # Check that the output looks like an LTO-IR container
    header = int.from_bytes(ltoir[:4], byteorder="little")
    if header != LTOIR_MAGIC:
        print(
            f"Unexpected header value 0x{header:X}.\n"
            f"Expected LTO-IR magic number 0x{LTOIR_MAGIC:X}."
            "\nExiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    return ltoir


def main(sourcepath, outputpath, arch):
    with open(sourcepath) as f:
        source = f.read()

    name = pathlib.Path(sourcepath).name
    ltoir = get_ltoir(source, name, arch)

    print(f"Writing {outputpath}...")

    with open(outputpath, "wb") as f:
        f.write(ltoir)


if __name__ == "__main__":
    description = "Compiles CUDA C/C++ to LTO-IR using NVRTC."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("sourcepath", help="path to source file")
    parser.add_argument(
        "-o", "--output", help="path to output file", default=None
    )
    parser.add_argument(
        "-a",
        "--arch",
        help="compute arch to target (e.g. sm_87). " "Defaults to sm_50.",
        default="sm_50",
    )

    args = parser.parse_args()
    outputpath = args.output

    if outputpath is None:
        outputpath = pathlib.Path(args.sourcepath).with_suffix(".ltoir")

    main(args.sourcepath, outputpath, args.arch)
