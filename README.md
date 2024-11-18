# Numba CUDA Target

An out-of-tree CUDA target for Numba.

This contains an entire copy of Numba's CUDA target (the `numba.cuda` module),
and a mechanism to ensure the code from this module (`numba_cuda.numba.cuda`) is
used as the `numba.cuda` module instead of the code from the `numba` package.

This is presently in an early state and is published for testing and feedback.

## Building / testing

Install as an editable install:

```
pip install -e .
```

Running tests:

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
