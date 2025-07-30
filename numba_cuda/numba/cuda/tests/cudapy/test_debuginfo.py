from numba.tests.support import override_config, captured_stdout
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest


@skip_on_cudasim("Simulator does not produce debug dumps")
class TestCudaDebugInfo(CUDATestCase):
    """
    These tests only checks the compiled PTX for debuginfo section
    """

    def _getasm(self, fn, sig):
        fn.compile(sig)
        return fn.inspect_asm(sig)

    def _check(self, fn, sig, expect):
        asm = self._getasm(fn, sig=sig)
        re_section_dbginfo = re.compile(r"\.section\s+\.debug_info\s+{")
        match = re_section_dbginfo.search(asm)
        assertfn = self.assertIsNotNone if expect else self.assertIsNone
        assertfn(match, msg=asm)

    def test_no_debuginfo_in_asm(self):
        @cuda.jit(debug=False)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(types.int32[:],), expect=False)

    def test_debuginfo_in_asm(self):
        @cuda.jit(debug=True, opt=False)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(types.int32[:],), expect=True)

    def test_environment_override(self):
        with override_config("CUDA_DEBUGINFO_DEFAULT", 1):
            # Using default value
            @cuda.jit(opt=False)
            def foo(x):
                x[0] = 1

            self._check(foo, sig=(types.int32[:],), expect=True)

            # User override default value
            @cuda.jit(debug=False)
            def bar(x):
                x[0] = 1

            self._check(bar, sig=(types.int32[:],), expect=False)

    def test_issue_5835(self):
        # Invalid debug metadata would segfault NVVM when any function was
        # compiled with debug turned on and optimization off. This eager
        # compilation should not crash anything.
        @cuda.jit((types.int32[::1],), debug=True, opt=False)
        def f(x):
            x[0] = 0

    def test_issue_9888(self):
        # Compiler created symbol should not be emitted in DILocalVariable
        # See Numba Issue #9888 https://github.com/numba/numba/pull/9888
        sig = (types.boolean,)

        @cuda.jit(sig, debug=True, opt=False)
        def f(cond):
            if cond:
                x = 1  # noqa: F841
            else:
                x = 0  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)
        # A varible name starting with "bool" in the debug metadata
        pat = r"!DILocalVariable\(.*name:\s+\"bool"
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNone(match, msg=llvm_ir)

    def test_bool_type(self):
        sig = (types.int32, types.int32)

        @cuda.jit("void(int32, int32)", debug=True, opt=False)
        def f(x, y):
            z = x == y  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)

        # extract the metadata node id from `type` field of DILocalVariable
        pat = r'!DILocalVariable\(.*name:\s+"z".*type:\s+!(\d+)'
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        mdnode_id = match.group(1)

        # verify the DIBasicType has correct encoding attribute DW_ATE_boolean
        pat = rf"!{mdnode_id}\s+=\s+!DIBasicType\(.*DW_ATE_boolean"
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)

    def test_grid_group_type(self):
        sig = (types.int32,)

        @cuda.jit(sig, debug=True, opt=False)
        def f(x):
            grid = cuda.cg.this_grid()  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)

        pat = r'!DIBasicType\(.*DW_ATE_unsigned, name: "GridGroup", size: 64'
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)

    @unittest.skip("Wrappers no longer exist")
    def test_wrapper_has_debuginfo(self):
        sig = (types.int32[::1],)

        @cuda.jit(sig, debug=True, opt=0)
        def f(x):
            x[0] = 1

        llvm_ir = f.inspect_llvm(sig)

        defines = [
            line
            for line in llvm_ir.splitlines()
            if 'define void @"_ZN6cudapy' in line
        ]

        # Make sure we only found one definition
        self.assertEqual(len(defines), 1)

        wrapper_define = defines[0]
        self.assertIn("!dbg", wrapper_define)

    def test_debug_function_calls_internal_impl(self):
        # Calling a function in a module generated from an implementation
        # internal to Numba requires multiple modules to be compiled with NVVM -
        # the internal implementation, and the caller. This example uses two
        # modules because the `in (2, 3)` is implemented with:
        #
        # numba::cpython::listobj::in_seq::$3clocals$3e::seq_contains_impl$242(
        #     UniTuple<long long, 2>,
        #     int
        # )
        #
        # This is condensed from this reproducer in Issue 5311:
        # https://github.com/numba/numba/issues/5311#issuecomment-674206587

        @cuda.jit((types.int32[:], types.int32[:]), debug=True, opt=False)
        def f(inp, outp):
            outp[0] = 1 if inp[0] in (2, 3) else 3

    def test_debug_function_calls_device_function(self):
        # Calling a device function requires compilation of multiple modules
        # with NVVM - one for the caller and one for the callee. This checks
        # that we don't cause an NVVM error in this case.

        @cuda.jit(device=True, debug=True, opt=0)
        def threadid():
            return cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

        @cuda.jit((types.int32[:],), debug=True, opt=0)
        def kernel(arr):
            i = cuda.grid(1)
            if i < len(arr):
                arr[i] = threadid()

    def _test_chained_device_function(self, kernel_debug, f1_debug, f2_debug):
        @cuda.jit(device=True, debug=f2_debug, opt=False)
        def f2(x):
            return x + 1

        @cuda.jit(device=True, debug=f1_debug, opt=False)
        def f1(x, y):
            return x - f2(y)

        @cuda.jit((types.int32, types.int32), debug=kernel_debug, opt=False)
        def kernel(x, y):
            f1(x, y)

        kernel[1, 1](1, 2)

    def test_chained_device_function(self):
        # Calling a device function that calls another device function from a
        # kernel with should succeed regardless of which jit decorators have
        # debug=True. See Issue #7159.

        debug_opts = itertools.product(*[(True, False)] * 3)

        for kernel_debug, f1_debug, f2_debug in debug_opts:
            with self.subTest(
                kernel_debug=kernel_debug, f1_debug=f1_debug, f2_debug=f2_debug
            ):
                self._test_chained_device_function(
                    kernel_debug, f1_debug, f2_debug
                )

    def _test_chained_device_function_two_calls(
        self, kernel_debug, f1_debug, f2_debug
    ):
        @cuda.jit(device=True, debug=f2_debug, opt=False)
        def f2(x):
            return x + 1

        @cuda.jit(device=True, debug=f1_debug, opt=False)
        def f1(x, y):
            return x - f2(y)

        @cuda.jit(debug=kernel_debug, opt=False)
        def kernel(x, y):
            f1(x, y)
            f2(x)

        kernel[1, 1](1, 2)

    def test_chained_device_function_two_calls(self):
        # Calling a device function that calls a leaf device function from a
        # kernel, and calling the leaf device function from the kernel should
        # succeed, regardless of which jit decorators have debug=True. See
        # Issue #7159.

        debug_opts = itertools.product(*[(True, False)] * 3)

        for kernel_debug, f1_debug, f2_debug in debug_opts:
            with self.subTest(
                kernel_debug=kernel_debug, f1_debug=f1_debug, f2_debug=f2_debug
            ):
                self._test_chained_device_function_two_calls(
                    kernel_debug, f1_debug, f2_debug
                )

    def test_chained_device_three_functions(self):
        # Like test_chained_device_function, but with enough functions (three)
        # to ensure that the recursion visits all the way down the call tree
        # when fixing linkage of functions for debug.
        def three_device_fns(kernel_debug, leaf_debug):
            @cuda.jit(device=True, debug=leaf_debug, opt=False)
            def f3(x):
                return x * x

            @cuda.jit(device=True)
            def f2(x):
                return f3(x) + 1

            @cuda.jit(device=True)
            def f1(x, y):
                return x - f2(y)

            @cuda.jit(debug=kernel_debug, opt=False)
            def kernel(x, y):
                f1(x, y)

            kernel[1, 1](1, 2)

        # Check when debug on the kernel, on the leaf, and not on any function.
        three_device_fns(kernel_debug=True, leaf_debug=True)
        three_device_fns(kernel_debug=True, leaf_debug=False)
        three_device_fns(kernel_debug=False, leaf_debug=True)
        three_device_fns(kernel_debug=False, leaf_debug=False)

    def _test_kernel_args_types(self):
        sig = (types.int32, types.int32)

        @cuda.jit("void(int32, int32)", debug=True, opt=False)
        def f(x, y):
            z = x + y  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)

        # extract the metadata node id from `types` field of DISubroutineType
        pat = r"!DISubroutineType\(types:\s+!(\d+)\)"
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        mdnode_id = match.group(1)

        # extract the metadata node ids from the flexible node of types
        pat = rf"!{mdnode_id}\s+=\s+!{{\s+!(\d+),\s+!(\d+)\s+}}"
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        mdnode_id1 = match.group(1)
        mdnode_id2 = match.group(2)

        # verify each of the two metadata nodes match expected type
        pat = rf'!{mdnode_id1}\s+=\s+!DIBasicType\(.*DW_ATE_signed,\s+name:\s+"int32"'  # noqa: E501
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        pat = rf'!{mdnode_id2}\s+=\s+!DIBasicType\(.*DW_ATE_signed,\s+name:\s+"int32"'  # noqa: E501
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)

    def test_kernel_args_types(self):
        self._test_kernel_args_types()

    def test_kernel_args_types_dump(self):
        # see issue#135
        with override_config("DUMP_LLVM", 1):
            with captured_stdout():
                self._test_kernel_args_types()

    def test_kernel_args_names(self):
        sig = (types.int32,)

        @cuda.jit("void(int32)", debug=True, opt=False)
        def f(x):
            z = x  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)

        # Verify argument name is not prefixed with "arg."
        pat = r"define void @.*\(i32 %\"x\"\)"
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        pat = r"define void @.*\(i32 %\"arg\.x\"\)"
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNone(match, msg=llvm_ir)

    def test_llvm_dbg_value(self):
        sig = (types.int32, types.int32)

        @cuda.jit("void(int32, int32)", debug=True, opt=False)
        def f(x, y):
            z1 = x  # noqa: F841
            z2 = 100  # noqa: F841
            z3 = y  # noqa: F841
            z4 = True  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)
        # Verify the call to llvm.dbg.declare is replaced by llvm.dbg.value
        pat1 = r'call void @"llvm.dbg.declare"'
        match = re.compile(pat1).search(llvm_ir)
        self.assertIsNone(match, msg=llvm_ir)
        pat2 = r'call void @"llvm.dbg.value"'
        match = re.compile(pat2).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)

    def test_no_user_var_alias(self):
        sig = (types.int32, types.int32)

        @cuda.jit("void(int32, int32)", debug=True, opt=False)
        def f(x, y):
            z = x  # noqa: F841
            z = y  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)
        pat = r'!DILocalVariable.*name:\s+"z\$1".*'
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNone(match, msg=llvm_ir)

    def test_no_literal_type(self):
        sig = (types.int32,)

        @cuda.jit("void(int32)", debug=True, opt=False)
        def f(x):
            z = x  # noqa: F841
            z = 100  # noqa: F841
            z = True  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)
        pat = r'!DIBasicType.*name:\s+"Literal.*'
        match = re.compile(pat).search(llvm_ir)
        self.assertIsNone(match, msg=llvm_ir)

    def test_union_poly_types(self):
        sig = (types.int32, types.int32)

        @cuda.jit("void(int32, int32)", debug=True, opt=False)
        def f(x, y):
            foo = 100  # noqa: F841
            foo = 2.34  # noqa: F841
            foo = True  # noqa: F841
            foo = 200  # noqa: F841

        llvm_ir = f.inspect_llvm(sig)
        # Extract the type node id
        pat1 = r'!DILocalVariable\(.*name: "foo".*type: !(\d+)\)'
        match = re.compile(pat1).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        mdnode_id = match.group(1)
        # Verify the union type and extract the elements node id
        pat2 = rf"!{mdnode_id} = distinct !DICompositeType\(elements: !(\d+),.*size: 64, tag: DW_TAG_union_type\)"  # noqa: E501
        match = re.compile(pat2).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        mdnode_id = match.group(1)
        # Extract the member node ids
        pat3 = r"!{ !(\d+), !(\d+), !(\d+) }"
        match = re.compile(pat3).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        mdnode_id1 = match.group(1)
        mdnode_id2 = match.group(2)
        mdnode_id3 = match.group(3)
        # Verify the member nodes
        pat4 = rf'!{mdnode_id1} = !DIDerivedType(.*name: "_bool", size: 8, tag: DW_TAG_member)'  # noqa: E501
        match = re.compile(pat4).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        pat5 = rf'!{mdnode_id2} = !DIDerivedType(.*name: "_float64", size: 64, tag: DW_TAG_member)'  # noqa: E501
        match = re.compile(pat5).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)
        pat6 = rf'!{mdnode_id3} = !DIDerivedType(.*name: "_int64", size: 64, tag: DW_TAG_member)'  # noqa: E501
        match = re.compile(pat6).search(llvm_ir)
        self.assertIsNotNone(match, msg=llvm_ir)

    def test_DW_LANG(self):
        @cuda.jit(debug=True)
        def foo():
            """
            CHECK: distinct !DICompileUnit
            CHECK-SAME: emissionKind: FullDebug
            CHECK-SAME: isOptimized: true
            CHECK-SAME: language: DW_LANG_C_plus_plus
            CHECK-SAME: producer: "clang (Numba)"
            """
            pass

        foo[1, 1]()

        llvm_ir = foo.inspect_llvm()[tuple()]
        self.assertFileCheckMatches(llvm_ir, foo.__doc__)

    def test_DILocation(self):
        """Tests that DILocation information is reasonable.

        The kernel `foo` produces LLVM like:
        define function() {
        entry:
          alloca
          store 0 to alloca
          <arithmetic for doing the operations on b, c, d>
          setup for print
          branch
        other_labels:
        ... <elided>
        }

        The following checks that:
        * the alloca and store have no !dbg
        * the arithmetic occurs in the order defined and with !dbg
        * that the !dbg entries are monotonically increasing in value with
          source line number
        """
        sig = (types.float64,)

        @cuda.jit(sig, debug=True)
        def foo(a):
            """
            CHECK-LABEL: define void @{{.+}}foo
            CHECK: entry:

            CHECK: %[[VAL_0:.*]] = alloca double
            CHECK-NOT: !dbg
            CHECK: store double 0.0, double* %[[VAL_0]]
            CHECK-NOT: !dbg
            CHECK: %[[VAL_1:.*]] = alloca double
            CHECK-NOT: !dbg
            CHECK: store double 0.0, double* %[[VAL_1]]
            CHECK-NOT: !dbg
            CHECK: %[[VAL_2:.*]] = alloca double
            CHECK-NOT: !dbg
            CHECK: store double 0.0, double* %[[VAL_2]]
            CHECK-NOT: !dbg
            CHECK: %[[VAL_3:.*]] = alloca double
            CHECK-NOT: !dbg
            CHECK: store double 0.0, double* %[[VAL_3]]
            CHECK-NOT: !dbg
            CHECK: %[[VAL_4:.*]] = alloca double
            CHECK-NOT: !dbg
            CHECK: store double 0.0, double* %[[VAL_4]]
            CHECK-NOT: !dbg
            CHECK: %[[VAL_5:.*]] = alloca double
            CHECK-NOT: !dbg
            CHECK: store double 0.0, double* %[[VAL_5]]
            CHECK-NOT: !dbg
            CHECK: %[[VAL_6:.*]] = alloca i8*
            CHECK-NOT: !dbg
            CHECK: store i8* null, i8** %[[VAL_6]]
            CHECK-NOT: !dbg
            CHECK: %[[VAL_7:.*]] = alloca i8*
            CHECK-NOT: !dbg
            CHECK: store i8* null, i8** %[[VAL_7]]
            CHECK-NOT: !dbg

            CHECK: br label %"[[ENTRY:.+]]"
            CHECK-NOT: !dbg
            CHECK: [[ENTRY]]:

            CHECK: fadd{{.+}} !dbg ![[DBGADD:[0-9]+]]
            CHECK: fmul{{.+}} !dbg ![[DBGMUL:[0-9]+]]
            CHECK: fdiv{{.+}} !dbg ![[DBGDIV:[0-9]+]]

            CHECK: ![[DBGADD]] = !DILocation
            CHECK: ![[DBGMUL]] = !DILocation
            CHECK: ![[DBGDIV]] = !DILocation
            """
            b = a + 1.23
            c = b * 2.34
            a = b / c

        ir = foo.inspect_llvm()[sig]
        self.assertFileCheckMatches(ir, foo.__doc__)

    def test_DITypes(self):
        """Tests that DITypes are emitted for the types used in the kernel."""
        sig = (
            types.float32,
            types.float64,
            types.int8,
            types.int16,
            types.int32,
            types.int64,
            types.uint8,
            types.uint16,
            types.uint32,
            types.uint64,
            types.complex64,
            types.complex128,
        )

        @cuda.jit(sig, debug=True)
        def foo(a, b, c, d, e, f, g, h, i, j, k, l):
            """
            CHECK: [[DBG1:.+]] = !DIBasicType(encoding: DW_ATE_boolean, name: "i8", size: 8)
            CHECK: [[DBG2:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG1]]
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_pointer_type
            CHECK: [[DBG3:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG2]]
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_pointer_type
            CHECK: [[DBG4:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float32", size: 32)
            CHECK: [[DBG5:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float64", size: 64)
            CHECK: [[DBG6:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int8", size: 8)
            CHECK: [[DBG7:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int16", size: 16)
            CHECK: [[DBG8:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int32", size: 32)
            CHECK: [[DBG9:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int64", size: 64)
            CHECK: [[DBG10:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint8", size: 8)
            CHECK: [[DBG11:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint16", size: 16)
            CHECK: [[DBG12:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint32", size: 32)
            CHECK: [[DBG13:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint64", size: 64)
            CHECK: [[DBG14:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float", size: 32)
            CHECK: [[DBG15:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG14]]
            CHECK-SAME: name: "real"
            CHECK-SAME: offset: 0
            CHECK-SAME: size: 32
            CHECK-SAME: tag: DW_TAG_member
            CHECK: [[DBG16:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float", size: 32)
            CHECK: [[DBG17:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG16]]
            CHECK-SAME: name: "imag"
            CHECK-SAME: offset: 32
            CHECK-SAME: size: 32
            CHECK-SAME: tag: DW_TAG_member
            CHECK: [[DBG18:.+]] = !{ [[DBG15]], [[DBG17]] }
            CHECK: [[DBG19:.+]] = distinct !DICompositeType(
            CHECK-SAME: elements: [[DBG18]]
            CHECK-SAME: name: "complex64 ({float, float})"
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_structure_type
            CHECK: [[DBG20:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "double", size: 64)
            CHECK: [[DBG21:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG20]]
            CHECK-SAME: name: "real"
            CHECK-SAME: offset: 0
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK: [[DBG22:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "double", size: 64)
            CHECK: [[DBG23:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG22]]
            CHECK-SAME: name: "imag"
            CHECK-SAME: offset: 64
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK: [[DBG24:.+]] = !{ [[DBG21]], [[DBG23]] }
            CHECK: [[DBG25:.+]] = distinct !DICompositeType(
            CHECK-SAME: elements: [[DBG24]]
            CHECK-SAME: name: "complex128 ({double, double})"
            CHECK-SAME: size: 128
            CHECK-SAME: tag: DW_TAG_structure_type
            CHECK: [[DBG32:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float32", size: 32)
            CHECK: [[DBG33:.+]] = !DILocalVariable(
            CHECK-SAME: name: "a"
            CHECK-SAME: type: [[DBG32]]
            CHECK: [[DBG34:.+]] = !DIExpression()
            CHECK: [[DBG35:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float64", size: 64)
            CHECK: [[DBG36:.+]] = !DILocalVariable(
            CHECK-SAME: name: "b"
            CHECK-SAME: type: [[DBG35]]
            CHECK: [[DBG37:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int8", size: 8)
            CHECK: [[DBG38:.+]] = !DILocalVariable(
            CHECK-SAME: name: "c"
            CHECK-SAME: type: [[DBG37]]
            CHECK: [[DBG39:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int16", size: 16)
            CHECK: [[DBG40:.+]] = !DILocalVariable(
            CHECK-SAME: name: "d"
            CHECK-SAME: type: [[DBG39]]
            CHECK: [[DBG41:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int32", size: 32)
            CHECK: [[DBG42:.+]] = !DILocalVariable(
            CHECK-SAME: name: "e"
            CHECK-SAME: type: [[DBG41]]
            CHECK: [[DBG43:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int64", size: 64)
            CHECK: [[DBG44:.+]] = !DILocalVariable(
            CHECK-SAME: name: "f"
            CHECK-SAME: type: [[DBG43]]
            CHECK: [[DBG45:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint8", size: 8)
            CHECK: [[DBG46:.+]] = !DILocalVariable(
            CHECK-SAME: name: "g"
            CHECK-SAME: type: [[DBG45]]
            CHECK: [[DBG47:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint16", size: 16)
            CHECK: [[DBG48:.+]] = !DILocalVariable(
            CHECK-SAME: name: "h"
            CHECK-SAME: type: [[DBG47]]
            CHECK: [[DBG49:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint32", size: 32)
            CHECK: [[DBG50:.+]] = !DILocalVariable(
            CHECK-SAME: name: "i"
            CHECK-SAME: type: [[DBG49]]
            CHECK: [[DBG51:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "uint64", size: 64)
            CHECK: [[DBG52:.+]] = !DILocalVariable(
            CHECK-SAME: name: "j"
            CHECK-SAME: type: [[DBG51]]
            CHECK: [[DBG53:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float", size: 32)
            CHECK: [[DBG54:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG53]]
            CHECK-SAME: name: "real"
            CHECK-SAME: offset: 0
            CHECK-SAME: size: 32
            CHECK-SAME: tag: DW_TAG_member
            CHECK: [[DBG55:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float", size: 32)
            CHECK: [[DBG56:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG55]]
            CHECK-SAME: name: "imag"
            CHECK-SAME: offset: 32
            CHECK-SAME: size: 32
            CHECK-SAME: tag: DW_TAG_member
            CHECK: [[DBG57:.+]] = !{ [[DBG54]], [[DBG56]] }
            CHECK: [[DBG58:.+]] = distinct !DICompositeType(
            CHECK-SAME: elements: [[DBG57]]
            CHECK-SAME: name: "complex64 ({float, float})"
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_structure_type
            CHECK: [[DBG59:.+]] = !DILocalVariable(
            CHECK-SAME: name: "k"
            CHECK-SAME: type: [[DBG58]]
            CHECK: [[DBG60:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "double", size: 64)
            CHECK: [[DBG61:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG60]]
            CHECK-SAME: name: "real"
            CHECK-SAME: offset: 0
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK: [[DBG62:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "double", size: 64)
            CHECK: [[DBG63:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG62]]
            CHECK-SAME: name: "imag"
            CHECK-SAME: offset: 64
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK: [[DBG64:.+]] = !{ [[DBG61]], [[DBG63]] }
            CHECK: [[DBG65:.+]] = distinct !DICompositeType(
            CHECK-SAME: elements: [[DBG64]]
            CHECK-SAME: name: "complex128 ({double, double})"
            CHECK-SAME: size: 128
            CHECK-SAME: tag: DW_TAG_structure_type
            CHECK: [[DBG66:.+]] = !DILocalVariable(
            CHECK-SAME: name: "l"
            CHECK-SAME: type: [[DBG65]]
            """
            pass

        ir = foo.inspect_llvm()[sig]
        self.assertFileCheckMatches(ir, foo.__doc__)

    def test_arrays(self):
        """Tests that arrays are emitted as DIBasicType."""
        sig = (types.float32[::1],)

        @cuda.jit(sig, debug=True)
        def foo(a):
            """
            CHECK: distinct !DICompileUnit
            CHECK: distinct !DISubprogram
            CHECK: [[DBG127:.+]] = !DIBasicType(encoding: DW_ATE_boolean, name: "i8", size: 8)
            CHECK: [[DBG128:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG127]]
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_pointer_type
            CHECK-SAME: )
            CHECK: [[DBG129:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG128]]
            CHECK-SAME: name: "meminfo"
            CHECK-SAME: offset: 0
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK-SAME: )
            CHECK: [[DBG130:.+]] = !DIBasicType(encoding: DW_ATE_boolean, name: "i8", size: 8)
            CHECK: [[DBG131:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG130]]
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_pointer_type
            CHECK-SAME: )
            CHECK: [[DBG132:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG131]]
            CHECK-SAME: name: "parent"
            CHECK-SAME: offset: 64
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK-SAME: )
            CHECK: [[DBG133:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int64", size: 64)
            CHECK: [[DBG134:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG133]]
            CHECK-SAME: name: "nitems"
            CHECK-SAME: offset: 128
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK-SAME: )
            CHECK: [[DBG135:.+]] = !DIBasicType(encoding: DW_ATE_signed, name: "int64", size: 64)
            CHECK: [[DBG136:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG135]]
            CHECK-SAME: name: "itemsize"
            CHECK-SAME: offset: 192
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK-SAME: )
            CHECK: [[DBG137:.+]] = !DIBasicType(encoding: DW_ATE_float, name: "float32", size: 32)
            CHECK: [[DBG138:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG137]]
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_pointer_type
            CHECK-SAME: )
            CHECK: [[DBG139:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG138]]
            CHECK-SAME: name: "data"
            CHECK-SAME: offset: 256
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK-SAME: )
            CHECK: [[DBG140:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "i64", size: 64)
            CHECK: [[DBG141:.+]] = !DICompositeType(
            CHECK-SAME: baseType: [[DBG140]]
            CHECK-SAME: identifier: "[1 x i64]"
            CHECK-SAME: name: "UniTuple(int64 x 1) ([1 x i64])"
            CHECK-SAME: tag: DW_TAG_array_type
            CHECK-SAME: )
            CHECK: [[DBG142:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG141]]
            CHECK-SAME: name: "shape"
            CHECK-SAME: offset: 320
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK-SAME: )
            CHECK: [[DBG143:.+]] = !DIBasicType(encoding: DW_ATE_unsigned, name: "i64", size: 64)
            CHECK: [[DBG144:.+]] = !DICompositeType(
            CHECK-SAME: baseType: [[DBG143]]
            CHECK-SAME: identifier: "[1 x i64]"
            CHECK-SAME: name: "UniTuple(int64 x 1) ([1 x i64])"
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_array_type
            CHECK-SAME: )
            CHECK: [[DBG145:.+]] = !DIDerivedType(
            CHECK-SAME: baseType: [[DBG144]]
            CHECK-SAME: name: "strides"
            CHECK-SAME: offset: 384
            CHECK-SAME: size: 64
            CHECK-SAME: tag: DW_TAG_member
            CHECK-SAME: )
            CHECK: [[DBG146:.+]] = !{ [[DBG129]], [[DBG132]], [[DBG134]], [[DBG136]], [[DBG139]], [[DBG142]], [[DBG145]] }
            CHECK: [[DBG147:.+]] = distinct !DICompositeType(
            CHECK-SAME: elements: [[DBG146]]
            CHECK-SAME: identifier: "{i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]}"
            CHECK-SAME: name: "array(float32, 1d, C) ({i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]})"
            CHECK-SAME: size: 448
            CHECK-SAME: tag: DW_TAG_structure_type
            CHECK-SAME: )
            CHECK: !DILocalVariable(
            CHECK-SAME: name: "a"
            CHECK-SAME: type: [[DBG147]]
            CHECK-SAME: )
            """
            pass

        ir = foo.inspect_llvm()[sig]
        self.assertFileCheckMatches(ir, foo.__doc__)


if __name__ == "__main__":
    unittest.main()
