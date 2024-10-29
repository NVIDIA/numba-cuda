<div align="center"><img src="docs/source/_static/numba-green-icon-rgb.svg" width="200"/></div>

# Numba CUDA Target

The CUDA target for Numba. Please visit the [official
documentation](https://nvidia.github.io/numba-cuda) to get started!


To report issues or file feature requests, please use the [issue
tracker](https://github.com/NVIDIA/nvmath-python/issues).

To raise questions or initiate discussions, please use the [Numba Discourse
forum](https://numba.discourse.group).

## Building from source

Install as an editable install:

```
pip install -e .
```

## Running tests

```
python -m numba.runtests numba.cuda.tests
```

This should discover the`numba.cuda` module from the `numba_cuda` package. You
can check where `numba.cuda` files are being located by running

```
python -c "from numba import cuda; print(cuda.__file__)"
```

which will show a path like:

```
<path to numba-cuda repo>/numba_cuda/numba/cuda/__init__.py
```
