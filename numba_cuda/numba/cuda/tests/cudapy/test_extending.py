from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from llvmlite import ir

import numpy as np
import os
from numba import config, cuda, njit, types
from numba.extending import overload


class Interval:
    """
    A half-open interval on the real number line.
    """

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return "Interval(%f, %f)" % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo


@njit
def interval_width(interval):
    return interval.width


@njit
def sum_intervals(i, j):
    return Interval(i.lo + j.lo, i.hi + j.hi)


if not config.ENABLE_CUDASIM:
    from numba.core import cgutils
    from numba.core.extending import (
        lower_builtin,
        models,
        type_callable,
        typeof_impl,
    )
    from numba.core.typing.templates import AttributeTemplate
    from numba.cuda.cudadecl import registry as cuda_registry
    from numba.cuda.cudaimpl import lower_attr as cuda_lower_attr
    from numba.cuda.extending import (
        register_model,
        make_attribute_wrapper,
    )

    class IntervalType(types.Type):
        def __init__(self):
            super().__init__(name="Interval")

    interval_type = IntervalType()

    @typeof_impl.register(Interval)
    def typeof_interval(val, c):
        return interval_type

    @type_callable(Interval)
    def type_interval(context):
        def typer(lo, hi):
            if isinstance(lo, types.Float) and isinstance(hi, types.Float):
                return interval_type

        return typer

    @register_model(IntervalType)
    class IntervalModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [
                ("lo", types.float64),
                ("hi", types.float64),
            ]
            models.StructModel.__init__(self, dmm, fe_type, members)

    make_attribute_wrapper(IntervalType, "lo", "lo")
    make_attribute_wrapper(IntervalType, "hi", "hi")

    @lower_builtin(Interval, types.Float, types.Float)
    def impl_interval(context, builder, sig, args):
        typ = sig.return_type
        lo, hi = args
        interval = cgutils.create_struct_proxy(typ)(context, builder)
        interval.lo = lo
        interval.hi = hi
        return interval._getvalue()

    @cuda_registry.register_attr
    class Interval_attrs(AttributeTemplate):
        key = IntervalType

        def resolve_width(self, mod):
            return types.float64

    @cuda_lower_attr(IntervalType, "width")
    def cuda_Interval_width(context, builder, sig, arg):
        lo = builder.extract_value(arg, 0)
        hi = builder.extract_value(arg, 1)
        return builder.fsub(hi, lo)


@skip_on_cudasim("Extensions not supported in the simulator")
class TestExtending(CUDATestCase):
    def test_attributes(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = iv.lo
            r[1] = iv.hi

        x = np.asarray((1.5, 2.5))
        r = np.zeros_like(x)

        f[1, 1](r, x)

        np.testing.assert_equal(r, x)

    def test_property(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = iv.width

        x = np.asarray((1.5, 2.5))
        r = np.zeros(1)

        f[1, 1](r, x)

        np.testing.assert_allclose(r[0], x[1] - x[0])

    def test_extension_type_as_arg(self):
        @cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = interval_width(iv)

        x = np.asarray((1.5, 2.5))
        r = np.zeros(1)

        f[1, 1](r, x)

        np.testing.assert_allclose(r[0], x[1] - x[0])

    def test_extension_type_as_retvalue(self):
        @cuda.jit
        def f(r, x):
            iv1 = Interval(x[0], x[1])
            iv2 = Interval(x[2], x[3])
            iv_sum = sum_intervals(iv1, iv2)
            r[0] = iv_sum.lo
            r[1] = iv_sum.hi

        x = np.asarray((1.5, 2.5, 3.0, 4.0))
        r = np.zeros(2)

        f[1, 1](r, x)

        expected = np.asarray((x[0] + x[2], x[1] + x[3]))
        np.testing.assert_allclose(r, expected)


TEST_BIN_DIR = os.getenv("NUMBA_CUDA_TEST_BIN_DIR")
if TEST_BIN_DIR:
    test_device_functions_a = os.path.join(
        TEST_BIN_DIR, "test_device_functions.a"
    )
    test_device_functions_cubin = os.path.join(
        TEST_BIN_DIR, "test_device_functions.cubin"
    )
    test_device_functions_cu = os.path.join(
        TEST_BIN_DIR, "test_device_functions.cu"
    )
    test_device_functions_fatbin = os.path.join(
        TEST_BIN_DIR, "test_device_functions.fatbin"
    )
    test_device_functions_fatbin_multi = os.path.join(
        TEST_BIN_DIR, "test_device_functions_multi.fatbin"
    )
    test_device_functions_o = os.path.join(
        TEST_BIN_DIR, "test_device_functions.o"
    )
    test_device_functions_ptx = os.path.join(
        TEST_BIN_DIR, "test_device_functions.ptx"
    )
    test_device_functions_ltoir = os.path.join(
        TEST_BIN_DIR, "test_device_functions.ltoir"
    )


@skip_on_cudasim("Extensions not supported in the simulator")
class TestExtendingLinkage(CUDATestCase):
    def test_extension_adds_linkable_code(self):
        cuda_major_version = cuda.runtime.get_version()[0]

        if cuda_major_version < 12:
            self.skipTest("CUDA 12 required for linking in-memory data")

        files = (
            (test_device_functions_a, cuda.Archive),
            (test_device_functions_cubin, cuda.Cubin),
            (test_device_functions_cu, cuda.CUSource),
            (test_device_functions_fatbin, cuda.Fatbin),
            (test_device_functions_o, cuda.Object),
            (test_device_functions_ptx, cuda.PTXSource),
            (test_device_functions_ltoir, cuda.LTOIR),
        )

        lto = config.CUDA_ENABLE_PYNVJITLINK

        for path, ctor in files:
            if ctor == cuda.LTOIR and not lto:
                # Don't try to test with LTOIR if LTO is not enabled
                continue

            with open(path, "rb") as f:
                code_object = ctor(f.read())

            def external_add(x, y):
                return x + y

            @type_callable(external_add)
            def type_external_add(context):
                def typer(x, y):
                    if x == types.uint32 and y == types.uint32:
                        return types.uint32

                return typer

            @lower_builtin(external_add, types.uint32, types.uint32)
            def lower_external_add(context, builder, sig, args):
                context.active_code_library.add_linking_file(code_object)
                i32 = ir.IntType(32)
                fnty = ir.FunctionType(i32, [i32, i32])
                fn = cgutils.get_or_insert_function(
                    builder.module, fnty, "add_cabi"
                )
                return builder.call(fn, args)

            @cuda.jit(lto=lto)
            def use_external_add(r, x, y):
                r[0] = external_add(x[0], y[0])

            r = np.zeros(1, dtype=np.uint32)
            x = np.ones(1, dtype=np.uint32)
            y = np.ones(1, dtype=np.uint32) * 2

            use_external_add[1, 1](r, x, y)

            np.testing.assert_equal(r[0], 3)

            @cuda.jit(lto=lto)
            def use_external_add_device(x, y):
                return external_add(x, y)

            @cuda.jit(lto=lto)
            def use_external_add_kernel(r, x, y):
                r[0] = use_external_add_device(x[0], y[0])

            r = np.zeros(1, dtype=np.uint32)
            x = np.ones(1, dtype=np.uint32)
            y = np.ones(1, dtype=np.uint32) * 2

            use_external_add_kernel[1, 1](r, x, y)

            np.testing.assert_equal(r[0], 3)

    def test_linked_called_through_overload(self):
        cu_code = cuda.CUSource("""
            extern "C" __device__
            int bar(int *out, int a)
            {
              *out = a * 2;
              return 0;
            }
        """)

        bar = cuda.declare_device("bar", "int32(int32)", link=cu_code)

        def bar_call(val):
            pass

        @overload(bar_call, target="cuda")
        def ol_bar_call(a):
            return lambda a: bar(a)

        @cuda.jit("void(int32[::1], int32[::1])")
        def foo(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = bar_call(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        foo[1, 32](r, x)

        np.testing.assert_equal(r, x * 2)


if __name__ == "__main__":
    unittest.main()
