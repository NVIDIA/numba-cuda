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

## Branching strategy

Presently the `main` branch is being used to target the exact behavior of the
built-in CUDA target. New feature development and bug fixes should be applied to
`develop`. Once the `main` branch is widely tested and confirmed to work well as
a drop-in replacement for the built-in `numba.cuda`, the `develop` branch will
be merged in and new feature development will proceed on `main`.

### Current PR targets

- PRs related to replacing the built-in CUDA target's features should target
  `main`.
- PRs adding new features and bug fixes should target `develop`.

### Future PR targets

- In future, all PRs should target the `main` branch.
