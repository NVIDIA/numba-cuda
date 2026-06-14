..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

Free-threading
==============

Free-threaded CPython test coverage requires a free-threaded Python build,
for example Python 3.14t with the ``cp314t`` ABI tag. The regular test suite
contains free-threaded smoke tests that run automatically in such an
environment. The heavier stress tests are opt-in because they create many
threads and subprocesses and are intended for local or dedicated CI runs.

Run the free-threading stress tests with::

   $ PYTHON_GIL=0 NUMBA_CUDA_FT_STRESS=1 \
       python -m pytest -q --pyargs numba.cuda.tests.stress

``PYTHON_GIL=0`` keeps the GIL disabled for free-threaded CPython builds. The
stress tests also check the build metadata and skip when Python is not a
free-threaded build.

Stress test environment variables
---------------------------------

.. envvar:: NUMBA_CUDA_FT_STRESS

   Enables the opt-in free-threading stress tests when set to ``1``,
   ``true``, ``yes``, or ``on``. Without this variable, tests in
   ``numba.cuda.tests.stress.test_free_threading`` are skipped.

.. envvar:: NUMBA_CUDA_FT_STRESS_SECONDS

   Controls the duration, in seconds, of timed CUDA stress tests. The default
   is ``30`` seconds.

.. envvar:: NUMBA_CUDA_FT_STRESS_WORKERS

   Controls the thread count used by thread-pool stress tests. Defaults vary
   by test and are capped at the detected CPU count unless this variable is
   set explicitly.

.. envvar:: NUMBA_CUDA_FT_STRESS_PROCESSES

   Controls the subprocess count used by cache-concurrency stress tests.
   Defaults vary by test and are capped at the detected CPU count unless this
   variable is set explicitly.

.. envvar:: NUMBA_CUDA_FT_STRESS_ITERS

   Overrides iteration counts for loop-based stress tests. Defaults vary by
   test, for example fingerprinting, memoryview buffer helpers, and type
   conversion stress cases.

Suggested stress profiles
-------------------------

Use the defaults for a quick local run. On many-core systems, explicitly set
the worker and process counts to exercise concurrent dispatcher, cache,
driver, and helper-extension paths more aggressively, for example::

   $ PYTHON_GIL=0 NUMBA_CUDA_FT_STRESS=1 \
       NUMBA_CUDA_FT_STRESS_SECONDS=120 \
       NUMBA_CUDA_FT_STRESS_WORKERS=96 \
       NUMBA_CUDA_FT_STRESS_PROCESSES=24 \
       NUMBA_CUDA_FT_STRESS_ITERS=10000 \
       python -m pytest -q --pyargs numba.cuda.tests.stress

If a stress failure is hard to reproduce, increase
``NUMBA_CUDA_FT_STRESS_SECONDS`` for CUDA tests or
``NUMBA_CUDA_FT_STRESS_ITERS`` for loop-based no-CUDA tests.
