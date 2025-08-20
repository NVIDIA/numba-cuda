import multiprocessing
import os
import shutil
import unittest
import warnings
import sys
import stat
import subprocess

from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import (
    CUDATestCase,
    skip_on_cudasim,
    skip_unless_cc_60,
    skip_if_cudadevrt_missing,
    test_data_dir,
)
from numba.cuda.tests.support import (
    TestCase,
    temp_directory,
    import_dynamic,
)


class BaseCacheTest(TestCase):
    # The source file that will be copied
    usecases_file = None
    # Make sure this doesn't conflict with another module
    modname = None

    def setUp(self):
        self.tempdir = temp_directory("test_cache")
        sys.path.insert(0, self.tempdir)
        self.modfile = os.path.join(self.tempdir, self.modname + ".py")
        self.cache_dir = os.path.join(self.tempdir, "__pycache__")
        shutil.copy(self.usecases_file, self.modfile)
        os.chmod(self.modfile, stat.S_IREAD | stat.S_IWRITE)
        self.maxDiff = None

    def tearDown(self):
        sys.modules.pop(self.modname, None)
        sys.path.remove(self.tempdir)

    def import_module(self):
        # Import a fresh version of the test module.  All jitted functions
        # in the test module will start anew and load overloads from
        # the on-disk cache if possible.
        old = sys.modules.pop(self.modname, None)
        if old is not None:
            # Make sure cached bytecode is removed
            cached = [old.__cached__]
            for fn in cached:
                try:
                    os.unlink(fn)
                except FileNotFoundError:
                    pass
        mod = import_dynamic(self.modname)
        self.assertEqual(mod.__file__.rstrip("co"), self.modfile)
        return mod

    def cache_contents(self):
        try:
            return [
                fn
                for fn in os.listdir(self.cache_dir)
                if not fn.endswith((".pyc", ".pyo"))
            ]
        except FileNotFoundError:
            return []

    def get_cache_mtimes(self):
        return dict(
            (fn, os.path.getmtime(os.path.join(self.cache_dir, fn)))
            for fn in sorted(self.cache_contents())
        )

    def check_pycache(self, n):
        c = self.cache_contents()
        self.assertEqual(len(c), n, c)

    def dummy_test(self):
        pass


class DispatcherCacheUsecasesTest(BaseCacheTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "dispatcher_caching_test_fodder"

    def run_in_separate_process(self, *, envvars={}):
        # Cached functions can be run from a distinct process.
        # Also stresses issue #1603: uncached function calling cached function
        # shouldn't fail compiling.
        code = """if 1:
            import sys

            sys.path.insert(0, %(tempdir)r)
            mod = __import__(%(modname)r)
            mod.self_test()
            """ % dict(tempdir=self.tempdir, modname=self.modname)

        subp_env = os.environ.copy()
        subp_env.update(envvars)
        popen = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=subp_env,
        )
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError(
                "process failed with code %s: \n"
                "stdout follows\n%s\n"
                "stderr follows\n%s\n"
                % (popen.returncode, out.decode(), err.decode()),
            )

    def check_hits(self, func, hits, misses=None):
        st = func.stats
        self.assertEqual(sum(st.cache_hits.values()), hits, st.cache_hits)
        if misses is not None:
            self.assertEqual(
                sum(st.cache_misses.values()), misses, st.cache_misses
            )


def check_access_is_preventable():
    # This exists to check whether it is possible to prevent access to
    # a file/directory through the use of `chmod 500`. If a user has
    # elevated rights (e.g. root) then writes are likely to be possible
    # anyway. Tests that require functioning access prevention are
    # therefore skipped based on the result of this check.
    tempdir = temp_directory("test_cache")
    test_dir = os.path.join(tempdir, "writable_test")
    os.mkdir(test_dir)
    # check a write is possible
    with open(os.path.join(test_dir, "write_ok"), "wt") as f:
        f.write("check1")
    # now forbid access
    os.chmod(test_dir, 0o500)
    try:
        with open(os.path.join(test_dir, "write_forbidden"), "wt") as f:
            f.write("check2")
        # access prevention is not possible
        return False
    except PermissionError:
        # Check that the cause of the exception is due to access/permission
        # as per
        # https://github.com/conda/conda/blob/4.5.0/conda/gateways/disk/permissions.py#L35-L37  # noqa: E501
        # errno reports access/perm fail so access prevention via
        # `chmod 500` works for this user.
        return True
    finally:
        os.chmod(test_dir, 0o775)
        shutil.rmtree(test_dir)


_access_preventable = check_access_is_preventable()
_access_msg = "Cannot create a directory to which writes are preventable"
skip_bad_access = unittest.skipUnless(_access_preventable, _access_msg)


@skip_on_cudasim("Simulator does not implement caching")
class CUDACachingTest(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "cuda_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_caching(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)  # 1 index, 2 data
        self.check_hits(f.func, 0, 2)

        f = mod.record_return_aligned
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))

        f = mod.record_return_packed
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_pycache(6)  # 2 index, 4 data
        self.check_hits(f.func, 0, 2)

        # Check the code runs ok from another process
        self.run_in_separate_process()

    def test_no_caching(self):
        mod = self.import_module()

        f = mod.add_nocache_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(0)

    def test_many_locals(self):
        # Declaring many local arrays creates a very large LLVM IR, which
        # cannot be pickled due to the level of recursion it requires to
        # pickle. This test ensures that kernels with many locals (and
        # therefore large IR) can be cached. See Issue #8373:
        # https://github.com/numba/numba/issues/8373
        self.check_pycache(0)
        mod = self.import_module()
        f = mod.many_locals
        f[1, 1]()
        self.check_pycache(2)  # 1 index, 1 data

    def test_closure(self):
        mod = self.import_module()

        with warnings.catch_warnings():
            warnings.simplefilter("error", NumbaWarning)

            f = mod.closure1
            self.assertPreciseEqual(f(3), 6)  # 3 + 3 = 6
            f = mod.closure2
            self.assertPreciseEqual(f(3), 8)  # 3 + 5 = 8
            f = mod.closure3
            self.assertPreciseEqual(f(3), 10)  # 3 + 7 = 10
            f = mod.closure4
            self.assertPreciseEqual(f(3), 12)  # 3 + 9 = 12
            self.check_pycache(5)  # 1 nbi, 4 nbc

    def test_cache_reuse(self):
        mod = self.import_module()
        mod.add_usecase(2, 3)
        mod.add_usecase(2.5, 3.5)
        mod.outer_uncached(2, 3)
        mod.outer(2, 3)
        mod.record_return_packed(mod.packed_arr, 0)
        mod.record_return_aligned(mod.aligned_arr, 1)
        mod.simple_usecase_caller(2)
        mtimes = self.get_cache_mtimes()
        # Two signatures compiled
        self.check_hits(mod.add_usecase.func, 0, 2)

        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        f = mod2.add_usecase
        f(2, 3)
        self.check_hits(f.func, 1, 0)
        f(2.5, 3.5)
        self.check_hits(f.func, 2, 0)

        # The files haven't changed
        self.assertEqual(self.get_cache_mtimes(), mtimes)

        self.run_in_separate_process()
        self.assertEqual(self.get_cache_mtimes(), mtimes)

    def test_cache_invalidate(self):
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)

        # This should change the functions' results
        with open(self.modfile, "a") as f:
            f.write("\nZ = 10\n")

        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_recompile(self):
        # Explicit call to recompile() should overwrite the cache
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)

        mod = self.import_module()
        f = mod.add_usecase
        mod.Z = 10
        self.assertPreciseEqual(f(2, 3), 6)
        f.func.recompile()
        self.assertPreciseEqual(f(2, 3), 15)

        # Freshly recompiled version is re-used from other imports
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_same_names(self):
        # Function with the same names should still disambiguate
        mod = self.import_module()
        f = mod.renamed_function1
        self.assertPreciseEqual(f(2), 4)
        f = mod.renamed_function2
        self.assertPreciseEqual(f(2), 8)

    def _test_pycache_fallback(self):
        """
        With a disabled __pycache__, test there is a working fallback
        (e.g. on the user-wide cache dir)
        """
        mod = self.import_module()
        f = mod.add_usecase
        # Remove this function's cache files at the end, to avoid accumulation
        # across test calls.
        self.addCleanup(
            shutil.rmtree, f.func.stats.cache_path, ignore_errors=True
        )

        self.assertPreciseEqual(f(2, 3), 6)
        # It's a cache miss since the file was copied to a new temp location
        self.check_hits(f.func, 0, 1)

        # Test re-use
        mod2 = self.import_module()
        f = mod2.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_hits(f.func, 1, 0)

        # The __pycache__ is empty (otherwise the test's preconditions
        # wouldn't be met)
        self.check_pycache(0)

    @skip_bad_access
    @unittest.skipIf(
        os.name == "nt", "cannot easily make a directory read-only on Windows"
    )
    def test_non_creatable_pycache(self):
        # Make it impossible to create the __pycache__ directory
        old_perms = os.stat(self.tempdir).st_mode
        os.chmod(self.tempdir, 0o500)
        self.addCleanup(os.chmod, self.tempdir, old_perms)

        self._test_pycache_fallback()

    @skip_bad_access
    @unittest.skipIf(
        os.name == "nt", "cannot easily make a directory read-only on Windows"
    )
    def test_non_writable_pycache(self):
        # Make it impossible to write to the __pycache__ directory
        pycache = os.path.join(self.tempdir, "__pycache__")
        os.mkdir(pycache)
        old_perms = os.stat(pycache).st_mode
        os.chmod(pycache, 0o500)
        self.addCleanup(os.chmod, pycache, old_perms)

        self._test_pycache_fallback()

    def test_cannot_cache_linking_libraries(self):
        link = str(test_data_dir / "jitlink.ptx")
        msg = "Cannot pickle CUDACodeLibrary with linking files"
        with self.assertRaisesRegex(RuntimeError, msg):

            @cuda.jit("void()", cache=True, link=[link])
            def f():
                pass


@skip_on_cudasim("Simulator does not implement caching")
class CUDACooperativeGroupTest(DispatcherCacheUsecasesTest):
    # See Issue #9432: https://github.com/numba/numba/issues/9432
    # If a cached function using CG sync was the first thing to compile,
    # the compile would fail.
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cg_cache_usecases.py")
    modname = "cuda_cooperative_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    @skip_unless_cc_60
    @skip_if_cudadevrt_missing
    def test_cache_cg(self):
        # Functions using cooperative groups should be cacheable. See Issue
        # #8888: https://github.com/numba/numba/issues/8888
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        mod.cg_usecase(0)
        self.check_pycache(2)  # 1 index, 1 data

        # Check the code runs ok from another process
        self.run_in_separate_process()


@skip_on_cudasim("Simulator does not implement caching")
class CUDAAndCPUCachingTest(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_with_cpu_usecases.py")
    modname = "cuda_and_cpu_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_cpu_and_cuda_targets(self):
        # The same function jitted for CPU and CUDA targets should maintain
        # separate caches for each target.
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        f_cpu = mod.assign_cpu
        f_cuda = mod.assign_cuda
        self.assertPreciseEqual(f_cpu(5), 5)
        self.check_pycache(2)  # 1 index, 1 data
        self.assertPreciseEqual(f_cuda(5), 5)
        self.check_pycache(3)  # 1 index, 2 data

        self.check_hits(f_cpu.func, 0, 1)
        self.check_hits(f_cuda.func, 0, 1)

        self.assertPreciseEqual(f_cpu(5.5), 5.5)
        self.check_pycache(4)  # 1 index, 3 data
        self.assertPreciseEqual(f_cuda(5.5), 5.5)
        self.check_pycache(5)  # 1 index, 4 data

        self.check_hits(f_cpu.func, 0, 2)
        self.check_hits(f_cuda.func, 0, 2)

    def test_cpu_and_cuda_reuse(self):
        # Existing cache files for the CPU and CUDA targets are reused.
        mod = self.import_module()
        mod.assign_cpu(5)
        mod.assign_cpu(5.5)
        mod.assign_cuda(5)
        mod.assign_cuda(5.5)

        mtimes = self.get_cache_mtimes()

        # Two signatures compiled
        self.check_hits(mod.assign_cpu.func, 0, 2)
        self.check_hits(mod.assign_cuda.func, 0, 2)

        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        f_cpu = mod2.assign_cpu
        f_cuda = mod2.assign_cuda

        f_cpu(2)
        self.check_hits(f_cpu.func, 1, 0)
        f_cpu(2.5)
        self.check_hits(f_cpu.func, 2, 0)
        f_cuda(2)
        self.check_hits(f_cuda.func, 1, 0)
        f_cuda(2.5)
        self.check_hits(f_cuda.func, 2, 0)

        # The files haven't changed
        self.assertEqual(self.get_cache_mtimes(), mtimes)

        self.run_in_separate_process()
        self.assertEqual(self.get_cache_mtimes(), mtimes)


def get_different_cc_gpus():
    # Find two GPUs with different Compute Capabilities and return them as a
    # tuple. If two GPUs with distinct Compute Capabilities cannot be found,
    # then None is returned.
    first_gpu = cuda.gpus[0]
    with first_gpu:
        first_cc = cuda.current_context().device.compute_capability

    for gpu in cuda.gpus[1:]:
        with gpu:
            cc = cuda.current_context().device.compute_capability
            if cc != first_cc:
                return (first_gpu, gpu)

    return None


@skip_on_cudasim("Simulator does not implement caching")
class TestMultiCCCaching(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "cuda_multi_cc_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_cache(self):
        gpus = get_different_cc_gpus()
        if not gpus:
            self.skipTest("Need two different CCs for multi-CC cache test")

        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)

        # Step 1. Populate the cache with the first GPU
        with gpus[0]:
            f = mod.add_usecase
            self.assertPreciseEqual(f(2, 3), 6)
            self.check_pycache(2)  # 1 index, 1 data
            self.assertPreciseEqual(f(2.5, 3), 6.5)
            self.check_pycache(3)  # 1 index, 2 data
            self.check_hits(f.func, 0, 2)

            f = mod.record_return_aligned
            rec = f(mod.aligned_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

            f = mod.record_return_packed
            rec = f(mod.packed_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))
            self.check_pycache(6)  # 2 index, 4 data
            self.check_hits(f.func, 0, 2)

        # Step 2. Run with the second GPU - under present behaviour this
        # doesn't further populate the cache.
        with gpus[1]:
            f = mod.add_usecase
            self.assertPreciseEqual(f(2, 3), 6)
            self.check_pycache(6)  # cache unchanged
            self.assertPreciseEqual(f(2.5, 3), 6.5)
            self.check_pycache(6)  # cache unchanged
            self.check_hits(f.func, 0, 2)

            f = mod.record_return_aligned
            rec = f(mod.aligned_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

            f = mod.record_return_packed
            rec = f(mod.packed_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))
            self.check_pycache(6)  # cache unchanged
            self.check_hits(f.func, 0, 2)

        # Step 3. Run in a separate module with the second GPU - this populates
        # the cache for the second CC.
        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)

        with gpus[1]:
            f = mod2.add_usecase
            self.assertPreciseEqual(f(2, 3), 6)
            self.check_pycache(7)  # 2 index, 5 data
            self.assertPreciseEqual(f(2.5, 3), 6.5)
            self.check_pycache(8)  # 2 index, 6 data
            self.check_hits(f.func, 0, 2)

            f = mod2.record_return_aligned
            rec = f(mod.aligned_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

            f = mod2.record_return_packed
            rec = f(mod.packed_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))
            self.check_pycache(10)  # 2 index, 8 data
            self.check_hits(f.func, 0, 2)

        # The following steps check that we can use the NVVM IR loaded from the
        # cache to generate PTX for a different compute capability to the
        # cached cubin's CC. To check this, we create another module that loads
        # the cached version containing a cubin for GPU 1. There will be no
        # cubin for GPU 0, so when we try to use it the PTX must be generated.

        mod3 = self.import_module()
        self.assertIsNot(mod, mod3)

        # Step 4. Run with GPU 1 and get a cache hit, loading the cache created
        # during Step 3.
        with gpus[1]:
            f = mod3.add_usecase
            self.assertPreciseEqual(f(2, 3), 6)
            self.assertPreciseEqual(f(2.5, 3), 6.5)

            f = mod3.record_return_aligned
            rec = f(mod.aligned_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

            f = mod3.record_return_packed
            rec = f(mod.packed_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

        # Step 5. Run with GPU 0 using the module from Step 4, to force PTX
        # generation from cached NVVM IR.
        with gpus[0]:
            f = mod3.add_usecase
            self.assertPreciseEqual(f(2, 3), 6)
            self.assertPreciseEqual(f(2.5, 3), 6.5)

            f = mod3.record_return_aligned
            rec = f(mod.aligned_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))

            f = mod3.record_return_packed
            rec = f(mod.packed_arr, 1)
            self.assertPreciseEqual(tuple(rec), (2, 43.5))


def child_initializer():
    # Disable occupancy and implicit copy warnings in processes in a
    # multiprocessing pool.
    from numba.core import config

    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    config.CUDA_WARN_ON_IMPLICIT_COPY = 0


@skip_on_cudasim("Simulator does not implement caching")
class TestMultiprocessCache(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, "cache_usecases.py")
    modname = "cuda_mp_caching_test_fodder"

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_multiprocessing(self):
        # Check caching works from multiple processes at once (#2028)
        mod = self.import_module()
        # Calling a pure Python caller of the JIT-compiled function is
        # necessary to reproduce the issue.
        f = mod.simple_usecase_caller
        n = 3
        try:
            ctx = multiprocessing.get_context("spawn")
        except AttributeError:
            ctx = multiprocessing

        pool = ctx.Pool(n, child_initializer)

        try:
            res = sum(pool.imap(f, range(n)))
        finally:
            pool.close()
        self.assertEqual(res, n * (n - 1) // 2)


@skip_on_cudasim("Simulator does not implement the CUDACodeLibrary")
class TestCUDACodeLibrary(CUDATestCase):
    # For tests of miscellaneous CUDACodeLibrary behaviour that we wish to
    # explicitly check

    def test_cannot_serialize_unfinalized(self):
        # The CUDA codegen failes to import under the simulator, so we cannot
        # import it at the top level
        from numba.cuda.codegen import CUDACodeLibrary

        # Usually a CodeLibrary requires a real CodeGen, but since we don't
        # interact with it, anything will do
        codegen = object()
        name = "library"
        cl = CUDACodeLibrary(codegen, name)
        with self.assertRaisesRegex(RuntimeError, "Cannot pickle unfinalized"):
            cl._reduce_states()
