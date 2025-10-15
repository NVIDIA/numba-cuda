<div align="center"><img src="docs/source/_static/numba-green-icon-rgb.svg" width="200"/></div>

# Numba CUDA Target

The CUDA target for Numba. Please visit the [official
documentation](https://nvidia.github.io/numba-cuda) to get started!


To report issues or file feature requests, please use the [issue
tracker](https://github.com/NVIDIA/numba-cuda/issues).

To raise questions or initiate discussions, please use the [Numba Discourse
forum](https://numba.discourse.group).

## Installation with pip or conda

Please refer to the [Installation documentation](https://nvidia.github.io/numba-cuda/user/installation.html#installation-with-a-python-package-manager).


## Installation from source

Install as an editable install:

```
pip install -e .
```

If you want to manage all run-time dependencies yourself, also pass the `--no-deps` flag.

## Running tests

Tests must be run from the `testing` folder, which contains the pytest
configuration and code to generate binaries used during the tests. The test
binaries need to be built on the system on which the tests are run, so that
they are compiled for the appropriate compute capability.

```
cd testing
# Optionally, build test binaries and point to their location for the test suite
make -j $(nproc)
export NUMBA_CUDA_TEST_BIN_DIR=`pwd`
# Execute tests
pytest -n auto -v
```

Alternatively, you can use [pixi](https://pixi.sh/latest/installation/) to wrap all of that up for you:

```
# run tests against CUDA 13
pixi run -e cu13 test -n auto -v
```


Testing should discover the `numba.cuda` module from the `numba_cuda` package. You
can check where `numba.cuda` files are being located by running

```
python -c "from numba import cuda; print(cuda.__file__)"
```

which will show a path like:

```
<path to numba-cuda repo>/numba_cuda/numba/cuda/__init__.py
```

## Contributing Guide

Review the
[CONTRIBUTING.md](https://github.com/NVIDIA/numba-cuda/blob/main/CONTRIBUTING.md)
file for information on how to contribute code and issues to the project.
