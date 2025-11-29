# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# Tests numba.analysis functions
import collections
import types as pytypes

import numpy as np
from numba.cuda.compiler import run_frontend
from numba.cuda.flags import Flags
from numba.cuda.core.compiler import StateDict
from numba.cuda import jit
from numba.cuda import types
from numba.cuda.core import errors
from numba.cuda.core import ir
from numba.cuda.utils import PYVERSION
from numba.cuda.core import postproc, rewrites, ir_utils
from numba.cuda.core.options import ParallelOptions
from numba.cuda.core.inline_closurecall import InlineClosureCallPass
from numba.cuda.tests.support import TestCase, override_config
from numba.cuda.core.analysis import (
    dead_branch_prune,
    rewrite_semantic_constants,
)
from numba.cuda.core.untyped_passes import (
    ReconstructSSA,
)
import unittest
from numba.cuda.core import config

_GLOBAL = 123

enable_pyobj_flags = Flags()
enable_pyobj_flags.enable_pyobject = True

if config.ENABLE_CUDASIM:
    raise unittest.SkipTest("Analysis passes not done in simulator")


def compile_to_ir(func):
    func_ir = run_frontend(func)
    state = StateDict()
    state.func_ir = func_ir
    state.typemap = None
    state.calltypes = None
    # Transform to SSA
    ReconstructSSA().run_pass(state)
    # call this to get print etc rewrites
    rewrites.rewrite_registry.apply("before-inference", state)
    return func_ir


class TestBranchPruneBase(TestCase):
    """
    Tests branch pruning
    """

    _DEBUG = False

    # find *all* branches
    def find_branches(self, the_ir):
        branches = []
        for blk in the_ir.blocks.values():
            tmp = [_ for _ in blk.find_insts(cls=ir.Branch)]
            branches.extend(tmp)
        return branches

    def assert_prune(self, func, args_tys, prune, *args, **kwargs):
        # This checks that the expected pruned branches have indeed been pruned.
        # func is a python function to assess
        # args_tys is the numba types arguments tuple
        # prune arg is a list, one entry per branch. The value in the entry is
        # encoded as follows:
        # True: using constant inference only, the True branch will be pruned
        # False: using constant inference only, the False branch will be pruned
        # None: under no circumstances should this branch be pruned
        # *args: the argument instances to pass to the function to check
        #        execution is still valid post transform
        # **kwargs:
        #        - flags: args to pass to `jit` default is `nopython=True`,
        #          e.g. permits use of e.g. object mode.

        func_ir = compile_to_ir(func)
        before = func_ir.copy()
        if self._DEBUG:
            print("=" * 80)
            print("before inline")
            func_ir.dump()

        # run closure inlining to ensure that nonlocals in closures are visible
        inline_pass = InlineClosureCallPass(
            func_ir,
            ParallelOptions(False),
        )
        inline_pass.run()

        # Remove all Dels, and re-run postproc
        post_proc = postproc.PostProcessor(func_ir)
        post_proc.run()

        rewrite_semantic_constants(func_ir, args_tys)
        if self._DEBUG:
            print("=" * 80)
            print("before prune")
            func_ir.dump()

        dead_branch_prune(func_ir, args_tys)

        after = func_ir
        if self._DEBUG:
            print("after prune")
            func_ir.dump()

        before_branches = self.find_branches(before)
        self.assertEqual(len(before_branches), len(prune))

        # what is expected to be pruned
        expect_removed = []
        for idx, prune in enumerate(prune):
            branch = before_branches[idx]
            if prune is True:
                expect_removed.append(branch.truebr)
            elif prune is False:
                expect_removed.append(branch.falsebr)
            elif prune is None:
                pass  # nothing should be removed!
            elif prune == "both":
                expect_removed.append(branch.falsebr)
                expect_removed.append(branch.truebr)
            else:
                assert 0, "unreachable"

        # compare labels
        original_labels = set([_ for _ in before.blocks.keys()])
        new_labels = set([_ for _ in after.blocks.keys()])
        # assert that the new labels are precisely the original less the
        # expected pruned labels
        try:
            self.assertEqual(new_labels, original_labels - set(expect_removed))
        except AssertionError as e:
            print("new_labels", sorted(new_labels))
            print("original_labels", sorted(original_labels))
            print("expect_removed", sorted(expect_removed))
            raise e

        if [
            arg is types.NoneType("none") or arg is types.Omitted(None)
            for arg in args_tys
        ].count(True) == 0:
            self.run_func(func, args)

    def run_func(self, impl, args):
        cres = jit(impl)
        dargs = args
        out = np.zeros(1)
        cout = np.zeros(1)
        args += (out,)
        dargs += (cout,)
        cres.py_func(*args)
        with override_config("DISABLE_PERFORMANCE_WARNINGS", 1):
            cres[1, 1](*dargs)
        self.assertPreciseEqual(out[0], cout[0])


class TestBranchPrune(TestBranchPruneBase):
    def test_single_if(self):
        def impl(x, res):
            if 1 == 0:
                res[0] = 3.14159

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [True],
            None,
        )

        def impl(x, res):
            if 1 == 1:
                res[0] = 3.14159

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [False],
            None,
        )

        def impl(x, res):
            if x is None:
                res[0] = 3.14159

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [False],
            None,
        )
        self.assert_prune(
            impl,
            (types.IntegerLiteral(10), types.Array(types.float64, 1, "C")),
            [True],
            10,
        )

        def impl(x, res):
            if x == 10:
                res[0] = 3.14159

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [True],
            None,
        )
        self.assert_prune(
            impl,
            (types.IntegerLiteral(10), types.Array(types.float64, 1, "C")),
            [None],
            10,
        )

        def impl(x, res):
            if x == 10:
                z = 3.14159  # noqa: F841 # no effect

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [True],
            None,
        )
        self.assert_prune(
            impl,
            (types.IntegerLiteral(10), types.Array(types.float64, 1, "C")),
            [None],
            10,
        )

        def impl(x, res):
            z = None
            y = z
            if x == y:
                res[0] = 100

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [False],
            None,
        )
        self.assert_prune(
            impl,
            (types.IntegerLiteral(10), types.Array(types.float64, 1, "C")),
            [True],
            10,
        )

    def test_single_if_else(self):
        def impl(x, res):
            if x is None:
                res[0] = 3.14159
            else:
                res[0] = 1.61803

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [False],
            None,
        )
        self.assert_prune(
            impl,
            (types.IntegerLiteral(10), types.Array(types.float64, 1, "C")),
            [True],
            10,
        )

    def test_single_if_const_val(self):
        def impl(x, res):
            if x == 100:
                res[0] = 3.14159

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [True],
            None,
        )
        self.assert_prune(
            impl,
            (types.IntegerLiteral(100), types.Array(types.float64, 1, "C")),
            [None],
            100,
        )

        def impl(x, res):
            # switch the condition order
            if 100 == x:
                res[0] = 3.14159

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [True],
            None,
        )
        self.assert_prune(
            impl,
            (types.IntegerLiteral(100), types.Array(types.float64, 1, "C")),
            [None],
            100,
        )

    def test_single_if_else_two_const_val(self):
        def impl(x, y, res):
            if x == y:
                res[0] = 3.14159
            else:
                res[0] = 1.61803

        self.assert_prune(
            impl,
            (types.IntegerLiteral(100),) * 2
            + (types.Array(types.float64, 1, "C"),),
            [None],
            100,
            100,
        )
        self.assert_prune(
            impl,
            (types.NoneType("none"),) * 2
            + (types.Array(types.float64, 1, "C"),),
            [False],
            None,
            None,
        )
        self.assert_prune(
            impl,
            (
                types.IntegerLiteral(100),
                types.NoneType("none"),
                types.Array(types.float64, 1, "C"),
            ),
            [True],
            100,
            None,
        )
        self.assert_prune(
            impl,
            (
                types.IntegerLiteral(100),
                types.IntegerLiteral(1000),
                types.Array(types.float64, 1, "C"),
            ),
            [None],
            100,
            1000,
        )

    def test_single_if_else_w_following_undetermined(self):
        def impl(x, res):
            x_is_none_work = False
            if x is None:
                x_is_none_work = True
            else:
                dead = 7  # noqa: F841 # no effect

            if x_is_none_work:
                y = 10
            else:
                y = -3
            res[0] = y

        self.assert_prune(
            impl,
            (types.NoneType("none"), types.Array(types.float64, 1, "C")),
            [False, None],
            None,
        )
        self.assert_prune(
            impl,
            (types.IntegerLiteral(10), types.Array(types.float64, 1, "C")),
            [True, None],
            10,
        )

        def impl(x, res):
            x_is_none_work = False
            if x is None:
                x_is_none_work = True
            else:
                pass

            if x_is_none_work:
                y = 10
            else:
                y = -3
            res[0] = y

        # Python 3.10 creates a block with a NOP in it for the `pass` which
        # means it gets pruned.
        if PYVERSION >= (3, 10):
            # Python 3.10 creates a block with a NOP in it for the `pass` which
            # means it gets pruned.
            self.assert_prune(
                impl,
                (types.NoneType("none"), types.Array(types.float64, 1, "C")),
                [False, None],
                None,
            )
        else:
            self.assert_prune(
                impl,
                (types.NoneType("none"), types.Array(types.float64, 1, "C")),
                [None, None],
                None,
            )

        self.assert_prune(
            impl,
            (types.IntegerLiteral(10), types.Array(types.float64, 1, "C")),
            [True, None],
            10,
        )

    def test_double_if_else_rt_const(self):
        def impl(x, res):
            one_hundred = 100
            x_is_none_work = 4
            if x is None:
                x_is_none_work = 100
            else:
                dead = 7  # noqa: F841 # no effect

            if x_is_none_work == one_hundred:
                y = 10
            else:
                y = -3

            res[0] = y + x_is_none_work

        self.assert_prune(impl, (types.NoneType("none"),), [False, None], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)

    def test_double_if_else_non_literal_const(self):
        def impl(x, res):
            one_hundred = 100
            if x == one_hundred:
                res[0] = 3.14159
            else:
                res[0] = 1.61803

        # no prune as compilation specialization on literal value not permitted
        self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)
        self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)

    def test_single_two_branches_same_cond(self):
        def impl(x, res):
            if x is None:
                y = 10
            else:
                y = 40

            if x is not None:
                z = 100
            else:
                z = 400

            res[0] = z + y

        self.assert_prune(impl, (types.NoneType("none"),), [False, True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, False], 10)

    def test_cond_is_kwarg_none(self):
        def impl(x=None, res=None):
            if x is None:
                y = 10
            else:
                y = 40

            if x is not None:
                z = 100
            else:
                z = 400

            res[0] = z + y

        self.assert_prune(impl, (types.Omitted(None),), [False, True], None)
        self.assert_prune(impl, (types.NoneType("none"),), [False, True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, False], 10)

    def test_cond_is_kwarg_value(self):
        def impl(x=1000, res=None):
            if x == 1000:
                y = 10
            else:
                y = 40

            if x != 1000:
                z = 100
            else:
                z = 400

            res[0] = z + y

        self.assert_prune(impl, (types.Omitted(1000),), [None, None], 1000)
        self.assert_prune(
            impl, (types.IntegerLiteral(1000),), [None, None], 1000
        )
        self.assert_prune(impl, (types.IntegerLiteral(0),), [None, None], 0)
        self.assert_prune(impl, (types.NoneType("none"),), [True, False], None)

    def test_cond_rewrite_is_correct(self):
        # this checks that when a condition is replaced, it is replace by a
        # true/false bit that correctly represents the evaluated condition
        def fn(x):
            if x is None:
                return 10
            return 12

        def check(func, arg_tys, bit_val):
            func_ir = compile_to_ir(func)

            # check there is 1 branch
            before_branches = self.find_branches(func_ir)
            self.assertEqual(len(before_branches), 1)

            # check the condition in the branch is a binop
            pred_var = before_branches[0].cond
            pred_defn = ir_utils.get_definition(func_ir, pred_var)
            self.assertEqual(pred_defn.op, "call")
            condition_var = pred_defn.args[0]
            condition_op = ir_utils.get_definition(func_ir, condition_var)
            self.assertEqual(condition_op.op, "binop")

            # do the prune, this should kill the dead branch and rewrite the
            #'condition to a true/false const bit
            if self._DEBUG:
                print("=" * 80)
                print("before prune")
                func_ir.dump()
            dead_branch_prune(func_ir, arg_tys)
            if self._DEBUG:
                print("=" * 80)
                print("after prune")
                func_ir.dump()

            # after mutation, the condition should be a const value `bit_val`
            new_condition_defn = ir_utils.get_definition(func_ir, condition_var)
            self.assertTrue(isinstance(new_condition_defn, ir.Const))
            self.assertEqual(new_condition_defn.value, bit_val)

        check(fn, (types.NoneType("none"),), 1)
        check(fn, (types.IntegerLiteral(10),), 0)

    def test_global_bake_in(self):
        def impl(x, res):
            if _GLOBAL == 123:
                res[0] = x
            else:
                res[0] = x + 10

        self.assert_prune(
            impl,
            (types.IntegerLiteral(1), types.Array(types.float64, 1, "C")),
            [False],
            1,
        )

        global _GLOBAL
        tmp = _GLOBAL

        try:
            _GLOBAL = 5

            def impl(x, res):
                if _GLOBAL == 123:
                    res[0] = x
                else:
                    res[0] = x + 10

            self.assert_prune(
                impl,
                (types.IntegerLiteral(1), types.Array(types.float64, 1, "C")),
                [True],
                1,
            )
        finally:
            _GLOBAL = tmp

    def test_freevar_bake_in(self):
        _FREEVAR = 123

        def impl(x, res):
            if _FREEVAR == 123:
                res[0] = x
            else:
                res[0] = x + 10

        self.assert_prune(
            impl,
            (types.IntegerLiteral(1), types.Array(types.float64, 1, "C")),
            [False],
            1,
        )

        _FREEVAR = 12

        def impl(x, res):
            if _FREEVAR == 123:
                res[0] = x
            else:
                res[0] = x + 10

        self.assert_prune(
            impl,
            (types.IntegerLiteral(1), types.Array(types.float64, 1, "C")),
            [True],
            1,
        )

    def test_redefined_variables_are_not_considered_in_prune(self):
        # see issue #4163, checks that if a variable that is an argument is
        # redefined in the user code it is not considered const

        def impl(array, a=None, res=None):
            if a is None:
                a = 0
            if a < 0:
                res[0] = 10
            res[0] = 30

        self.assert_prune(
            impl,
            (
                types.Array(types.float64, 2, "C"),
                types.NoneType("none"),
                types.Array(types.float64, 1, "C"),
            ),
            [None, None],
            np.zeros((2, 3)),
            None,
        )

    def test_redefinition_analysis_same_block(self):
        # checks that a redefinition in a block with prunable potential doesn't
        # break

        def impl(array, x, a=None, res=None):
            b = 2
            if x < 4:
                b = 12
            if a is None:  # known true
                a = 7  # live
            else:
                b = 15  # dead
            if a < 0:  # valid as a result of the redefinition of 'a'
                res[0] = 10
            res[0] = 30 + b + a

        self.assert_prune(
            impl,
            (
                types.Array(types.float64, 2, "C"),
                types.float64,
                types.NoneType("none"),
                types.Array(types.float64, 1, "C"),
            ),
            [None, False, None],
            np.zeros((2, 3)),
            1.0,
            None,
        )

    def test_redefinition_analysis_different_block_can_exec(self):
        # checks that a redefinition in a block that may be executed prevents
        # pruning

        def impl(array, x, res):
            b = 0
            if x > 5:
                a = 11  # a redefined, cannot tell statically if this will exec
            if x < 4:
                b = 12
            if a is None:  # cannot prune, cannot determine if re-defn occurred
                b += 5
            else:
                b += 7
                if a < 0:
                    res[0] = 10
            res[0] = 30 + b

        self.assert_prune(
            impl,
            (
                types.Array(types.float64, 2, "C"),
                types.float64,
                types.NoneType("none"),
                types.Array(types.float64, 1, "C"),
            ),
            [None, None, None, None],
            np.zeros((2, 3)),
            1.0,
            None,
        )

    def test_redefinition_analysis_different_block_cannot_exec(self):
        # checks that a redefinition in a block guarded by something that
        # has prune potential

        def impl(array, x=None, a=None, res=None):
            b = 0
            if x is not None:
                a = 11
            if a is None:
                b += 5
            else:
                b += 7
            res[0] = 30 + b

        self.assert_prune(
            impl,
            (
                types.Array(types.float64, 2, "C"),
                types.NoneType("none"),
                types.NoneType("none"),
                types.Array(types.float64, 1, "C"),
            ),
            [True, None],
            np.zeros((2, 3)),
            None,
            None,
        )

        self.assert_prune(
            impl,
            (
                types.Array(types.float64, 2, "C"),
                types.NoneType("none"),
                types.float64,
                types.Array(types.float64, 1, "C"),
            ),
            [True, None],
            np.zeros((2, 3)),
            None,
            1.2,
        )

        self.assert_prune(
            impl,
            (
                types.Array(types.float64, 2, "C"),
                types.float64,
                types.NoneType("none"),
                types.Array(types.float64, 1, "C"),
            ),
            [None, None],
            np.zeros((2, 3)),
            1.2,
            None,
        )

    def test_closure_and_nonlocal_can_prune(self):
        # Closures must be inlined ahead of branch pruning in case nonlocal
        # is used. See issue #6585.
        def impl(res):
            x = 1000

            def closure():
                nonlocal x
                x = 0

            closure()

            if x == 0:
                res[0] = True
            else:
                res[0] = False

        self.assert_prune(
            impl,
            (types.Array(types.float64, 1, "C"),),
            [
                False,
            ],
        )

    def test_closure_and_nonlocal_cannot_prune(self):
        # Closures must be inlined ahead of branch pruning in case nonlocal
        # is used. See issue #6585.
        def impl(n, res):
            x = 1000

            def closure(t):
                nonlocal x
                x = t

            closure(n)

            if x == 0:
                res[0] = True
            else:
                res[0] = False

        self.assert_prune(
            impl,
            (types.int64, types.Array(types.float64, 1, "C")),
            [
                None,
            ],
            1,
        )


class TestBranchPrunePredicates(TestBranchPruneBase):
    # Really important thing to remember... the branch on predicates end up as
    # POP_JUMP_IF_<bool> and the targets are backwards compared to normal, i.e.
    # the true condition is far jump and the false the near i.e. `if x` would
    # end up in Numba IR as e.g. `branch x 10, 6`.

    _TRUTHY = (1, "String", True, 7.4, 3j)
    _FALSEY = (0, "", False, 0.0, 0j, None)

    def _literal_const_sample_generator(self, pyfunc, consts):
        """
        This takes a python function, pyfunc, and manipulates its co_const
        __code__ member to create a new function with different co_consts as
        supplied in argument consts.

        consts is a dict {index: value} of co_const tuple index to constant
        value used to update a pyfunc clone's co_const.
        """
        pyfunc_code = pyfunc.__code__

        # translate consts spec to update the constants
        co_consts = {k: v for k, v in enumerate(pyfunc_code.co_consts)}
        for k, v in consts.items():
            co_consts[k] = v
        new_consts = tuple([v for _, v in sorted(co_consts.items())])

        # create code object with mutation
        new_code = pyfunc_code.replace(co_consts=new_consts)

        # get function
        return pytypes.FunctionType(new_code, globals())

    def test_literal_const_code_gen(self):
        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if _CONST1:
                return 3.14159
            else:
                _CONST2 = "PLACEHOLDER2"
            return _CONST2 + 4

        if PYVERSION in ((3, 14),):
            # The order of the __code__.co_consts changes with 3.14
            new = self._literal_const_sample_generator(impl, {0: 0, 2: 20})
        elif PYVERSION in ((3, 10), (3, 11), (3, 12), (3, 13)):
            new = self._literal_const_sample_generator(impl, {1: 0, 3: 20})
        else:
            raise NotImplementedError(PYVERSION)
        iconst = impl.__code__.co_consts
        nconst = new.__code__.co_consts
        if PYVERSION in ((3, 14),):
            self.assertEqual(iconst, ("PLACEHOLDER1", 3.14159, "PLACEHOLDER2"))
            self.assertEqual(nconst, (0, 3.14159, 20))
        elif PYVERSION in ((3, 10), (3, 11), (3, 12), (3, 13)):
            self.assertEqual(
                iconst, (None, "PLACEHOLDER1", 3.14159, "PLACEHOLDER2", 4)
            )
            self.assertEqual(nconst, (None, 0, 3.14159, 20, 4))
        else:
            raise NotImplementedError(PYVERSION)
        self.assertEqual(impl(None), 3.14159)
        self.assertEqual(new(None), 24)

    def test_single_if_const(self):
        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if _CONST1:
                return 3.14159

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:
                if PYVERSION in ((3, 14),):
                    # The order of the __code__.co_consts changes with 3.14
                    func = self._literal_const_sample_generator(
                        impl, {0: const}
                    )
                elif PYVERSION in ((3, 10), (3, 11), (3, 12), (3, 13)):
                    func = self._literal_const_sample_generator(
                        impl, {1: const}
                    )
                else:
                    raise NotImplementedError(PYVERSION)
                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_negate_const(self):
        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if not _CONST1:
                return 3.14159

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:
                if PYVERSION in ((3, 14),):
                    # The order of the __code__.co_consts changes with 3.14
                    func = self._literal_const_sample_generator(
                        impl, {0: const}
                    )
                elif PYVERSION in ((3, 10), (3, 11), (3, 12), (3, 13)):
                    func = self._literal_const_sample_generator(
                        impl, {1: const}
                    )
                else:
                    raise NotImplementedError(PYVERSION)
                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_else_const(self):
        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if _CONST1:
                return 3.14159
            else:
                return 1.61803

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:
                if PYVERSION in ((3, 14),):
                    # The order of the __code__.co_consts changes with 3.14
                    func = self._literal_const_sample_generator(
                        impl, {0: const}
                    )
                elif PYVERSION in ((3, 10), (3, 11), (3, 12), (3, 13)):
                    func = self._literal_const_sample_generator(
                        impl, {1: const}
                    )
                else:
                    raise NotImplementedError(PYVERSION)
                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_else_negate_const(self):
        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if not _CONST1:
                return 3.14159
            else:
                return 1.61803

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:
                if PYVERSION in ((3, 14),):
                    # The order of the __code__.co_consts changes with 3.14
                    func = self._literal_const_sample_generator(
                        impl, {0: const}
                    )
                elif PYVERSION in ((3, 10), (3, 11), (3, 12), (3, 13)):
                    func = self._literal_const_sample_generator(
                        impl, {1: const}
                    )
                else:
                    raise NotImplementedError(PYVERSION)
                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_freevar(self):
        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:

                def func(x):
                    if const:
                        return 3.14159, const

                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_negate_freevar(self):
        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:

                def func(x):
                    if not const:
                        return 3.14159, const

                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_else_negate_freevar(self):
        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:

                def func(x):
                    if not const:
                        return 3.14159, const
                    else:
                        return 1.61803, const

                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    # globals in this section have absurd names after their test usecase names
    # so as to prevent collisions and permit tests to run in parallel
    def test_single_if_global(self):
        global c_test_single_if_global

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for c in c_inp:
                c_test_single_if_global = c

                def func(x):
                    if c_test_single_if_global:
                        return 3.14159, c_test_single_if_global

                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_negate_global(self):
        global c_test_single_if_negate_global

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for c in c_inp:
                c_test_single_if_negate_global = c

                def func(x):
                    if c_test_single_if_negate_global:
                        return 3.14159, c_test_single_if_negate_global

                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_else_global(self):
        global c_test_single_if_else_global

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for c in c_inp:
                c_test_single_if_else_global = c

                def func(x):
                    if c_test_single_if_else_global:
                        return 3.14159, c_test_single_if_else_global
                    else:
                        return 1.61803, c_test_single_if_else_global

                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_single_if_else_negate_global(self):
        global c_test_single_if_else_negate_global

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for c in c_inp:
                c_test_single_if_else_negate_global = c

                def func(x):
                    if not c_test_single_if_else_negate_global:
                        return 3.14159, c_test_single_if_else_negate_global
                    else:
                        return 1.61803, c_test_single_if_else_negate_global

                self.assert_prune(
                    func, (types.NoneType("none"),), [prune], None
                )

    def test_issue_5618(self):
        @jit
        def foo(res):
            tmp = 666
            if tmp:
                res[0] = tmp

        self.run_func(foo, ())


class TestBranchPrunePostSemanticConstRewrites(TestBranchPruneBase):
    # Tests that semantic constants rewriting works by virtue of branch pruning

    def test_array_ndim_attr(self):
        def impl(array, res):
            if array.ndim == 2:
                if array.shape[1] == 2:
                    res[0] = 1
            else:
                res[0] = 10

        self.assert_prune(
            impl,
            (types.Array(types.float64, 2, "C"),),
            [False, None],
            np.zeros((2, 3)),
        )
        self.assert_prune(
            impl,
            (types.Array(types.float64, 1, "C"),),
            [True, "both"],
            np.zeros((2,)),
        )

    def test_tuple_len(self):
        def impl(tup, res):
            if len(tup) == 3:
                if tup[2] == 2:
                    res[0] = 1
            else:
                res[0] = 0

        self.assert_prune(
            impl,
            (types.UniTuple(types.int64, 3),),
            [False, None],
            tuple([1, 2, 3]),
        )
        self.assert_prune(
            impl,
            (types.UniTuple(types.int64, 2),),
            [True, "both"],
            tuple([1, 2]),
        )

    def test_attr_not_len(self):
        # The purpose of this test is to make sure that the conditions guarding
        # the rewrite part do not themselves raise exceptions.
        # This produces an `ir.Expr` call node for `float.as_integer_ratio`,
        # which is a getattr() on `float`.

        @jit
        def test():
            float.as_integer_ratio(1.23)

        # this should raise a TypingError
        with self.assertRaises(errors.TypingError) as e:
            test[1, 1]()

        self.assertIn("Unknown attribute 'as_integer_ratio'", str(e.exception))

    def test_ndim_not_on_array(self):
        FakeArray = collections.namedtuple("FakeArray", ["ndim"])
        fa = FakeArray(ndim=2)

        def impl(fa, res):
            if fa.ndim == 2:
                res[0] = fa.ndim

        # check prune works for array ndim
        self.assert_prune(
            impl,
            (types.Array(types.float64, 2, "C"),),
            [False],
            np.zeros((2, 3)),
        )

        # check prune fails for something with `ndim` attr that is not array
        FakeArrayType = types.NamedUniTuple(types.int64, 1, FakeArray)
        self.assert_prune(
            impl,
            (FakeArrayType,),
            [None],
            fa,
            flags={"nopython": False, "forceobj": True},
        )
