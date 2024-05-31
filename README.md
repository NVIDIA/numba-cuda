# Numba CUDA Target

An out-of-tree CUDA target for Numba.

This contains an entire copy of Numba's CUDA target (the `numba.cuda` module),
and a mechanism to ensure the code from this module (`numba_cuda.numba.cuda`) is
used as the `numba.cuda` module instead of the code from the `numba` package.

# Building / testing

Install as an editable install:

```
pip install -e .
```

The `_extras` library needs copying to the source tree for an editable install:

```
cp build/cp*/_extras.cpython-*.so numba_cuda/numba/cuda/cudadrv/
```

Run the test:

```
python test.py
```
