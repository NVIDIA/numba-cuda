# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from collections import namedtuple
from numba import types
from numba.core import consts, ir
from numba.core.analysis import compute_cfg_from_blocks


# Used to describe a nullified condition in dead branch pruning
nullified = namedtuple("nullified", "condition, taken_br, rewrite_stmt")


def dead_branch_prune(func_ir, called_args):
    """
    Removes dead branches based on constant inference from function args.
    This directly mutates the IR.

    func_ir is the IR
    called_args are the actual arguments with which the function is called
    """
    from numba.cuda.core.ir_utils import (
        get_definition,
        guard,
        find_const,
        GuardException,
    )

    DEBUG = 0

    def find_branches(func_ir):
        # find *all* branches
        branches = []
        for blk in func_ir.blocks.values():
            branch_or_jump = blk.body[-1]
            if isinstance(branch_or_jump, ir.Branch):
                branch = branch_or_jump
                pred = guard(get_definition, func_ir, branch.cond.name)
                if pred is not None and getattr(pred, "op", None) == "call":
                    function = guard(get_definition, func_ir, pred.func)
                    if (
                        function is not None
                        and isinstance(function, ir.Global)
                        and function.value is bool
                    ):
                        condition = guard(get_definition, func_ir, pred.args[0])
                        if condition is not None:
                            branches.append((branch, condition, blk))
        return branches

    def do_prune(take_truebr, blk):
        keep = branch.truebr if take_truebr else branch.falsebr
        # replace the branch with a direct jump
        jmp = ir.Jump(keep, loc=branch.loc)
        blk.body[-1] = jmp
        return 1 if keep == branch.truebr else 0

    def prune_by_type(branch, condition, blk, *conds):
        # this prunes a given branch and fixes up the IR
        # at least one needs to be a NoneType
        lhs_cond, rhs_cond = conds
        lhs_none = isinstance(lhs_cond, types.NoneType)
        rhs_none = isinstance(rhs_cond, types.NoneType)
        if lhs_none or rhs_none:
            try:
                take_truebr = condition.fn(lhs_cond, rhs_cond)
            except Exception:
                return False, None
            if DEBUG > 0:
                kill = branch.falsebr if take_truebr else branch.truebr
                print(
                    "Pruning %s" % kill,
                    branch,
                    lhs_cond,
                    rhs_cond,
                    condition.fn,
                )
            taken = do_prune(take_truebr, blk)
            return True, taken
        return False, None

    def prune_by_value(branch, condition, blk, *conds):
        lhs_cond, rhs_cond = conds
        try:
            take_truebr = condition.fn(lhs_cond, rhs_cond)
        except Exception:
            return False, None
        if DEBUG > 0:
            kill = branch.falsebr if take_truebr else branch.truebr
            print("Pruning %s" % kill, branch, lhs_cond, rhs_cond, condition.fn)
        do_prune(take_truebr, blk)
        # It is not safe to rewrite the predicate to a nominal value based on
        # which branch is taken, the rewritten const predicate needs to
        # hold the actual computed const value as something else may refer to
        # it!
        return True, take_truebr

    def prune_by_predicate(branch, pred, blk):
        try:
            # Just to prevent accidents, whilst already guarded, ensure this
            # is an ir.Const
            if not isinstance(pred, (ir.Const, ir.FreeVar, ir.Global)):
                raise TypeError("Expected constant Numba IR node")
            take_truebr = bool(pred.value)
        except TypeError:
            return False, None
        if DEBUG > 0:
            kill = branch.falsebr if take_truebr else branch.truebr
            print("Pruning %s" % kill, branch, pred)
        taken = do_prune(take_truebr, blk)
        return True, taken

    class Unknown(object):
        pass

    def resolve_input_arg_const(input_arg_idx):
        """
        Resolves an input arg to a constant (if possible)
        """
        input_arg_ty = called_args[input_arg_idx]

        # comparing to None?
        if isinstance(input_arg_ty, types.NoneType):
            return input_arg_ty

        # is it a kwarg default
        if isinstance(input_arg_ty, types.Omitted):
            val = input_arg_ty.value
            if isinstance(val, types.NoneType):
                return val
            elif val is None:
                return types.NoneType("none")

        # literal type, return the type itself so comparisons like `x == None`
        # still work as e.g. x = types.int64 will never be None/NoneType so
        # the branch can still be pruned
        return getattr(input_arg_ty, "literal_type", Unknown())

    if DEBUG > 1:
        print("before".center(80, "-"))
        print(func_ir.dump())

    phi2lbl = dict()
    phi2asgn = dict()
    for lbl, blk in func_ir.blocks.items():
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr) and stmt.value.op == "phi":
                    phi2lbl[stmt.value] = lbl
                    phi2asgn[stmt.value] = stmt

    # This looks for branches where:
    # at least one arg of the condition is in input args and const
    # at least one an arg of the condition is a const
    # if the condition is met it will replace the branch with a jump
    branch_info = find_branches(func_ir)
    # stores conditions that have no impact post prune
    nullified_conditions = []

    for branch, condition, blk in branch_info:
        const_conds = []
        if isinstance(condition, ir.Expr) and condition.op == "binop":
            prune = prune_by_value
            for arg in [condition.lhs, condition.rhs]:
                resolved_const = Unknown()
                arg_def = guard(get_definition, func_ir, arg)
                if isinstance(arg_def, ir.Arg):
                    # it's an e.g. literal argument to the function
                    resolved_const = resolve_input_arg_const(arg_def.index)
                    prune = prune_by_type
                else:
                    # it's some const argument to the function, cannot use guard
                    # here as the const itself may be None
                    try:
                        resolved_const = find_const(func_ir, arg)
                        if resolved_const is None:
                            resolved_const = types.NoneType("none")
                    except GuardException:
                        pass

                if not isinstance(resolved_const, Unknown):
                    const_conds.append(resolved_const)

            # lhs/rhs are consts
            if len(const_conds) == 2:
                # prune the branch, switch the branch for an unconditional jump
                prune_stat, taken = prune(branch, condition, blk, *const_conds)
                if prune_stat:
                    # add the condition to the list of nullified conditions
                    nullified_conditions.append(
                        nullified(condition, taken, True)
                    )
        else:
            # see if this is a branch on a constant value predicate
            resolved_const = Unknown()
            try:
                pred_call = get_definition(func_ir, branch.cond)
                resolved_const = find_const(func_ir, pred_call.args[0])
                if resolved_const is None:
                    resolved_const = types.NoneType("none")
            except GuardException:
                pass

            if not isinstance(resolved_const, Unknown):
                prune_stat, taken = prune_by_predicate(branch, condition, blk)
                if prune_stat:
                    # add the condition to the list of nullified conditions
                    nullified_conditions.append(
                        nullified(condition, taken, False)
                    )

    # 'ERE BE DRAGONS...
    # It is the evaluation of the condition expression that often trips up type
    # inference, so ideally it would be removed as it is effectively rendered
    # dead by the unconditional jump if a branch was pruned. However, there may
    # be references to the condition that exist in multiple places (e.g. dels)
    # and we cannot run DCE here as typing has not taken place to give enough
    # information to run DCE safely. Upshot of all this is the condition gets
    # rewritten below into a benign const that typing will be happy with and DCE
    # can remove it and its reference post typing when it is safe to do so
    # (if desired). It is required that the const is assigned a value that
    # indicates the branch taken as its mutated value would be read in the case
    # of object mode fall back in place of the condition itself. For
    # completeness the func_ir._definitions and ._consts are also updated to
    # make the IR state self consistent.

    deadcond = [x.condition for x in nullified_conditions]
    for _, cond, blk in branch_info:
        if cond in deadcond:
            for x in blk.body:
                if isinstance(x, ir.Assign) and x.value is cond:
                    # rewrite the condition as a true/false bit
                    nullified_info = nullified_conditions[deadcond.index(cond)]
                    # only do a rewrite of conditions, predicates need to retain
                    # their value as they may be used later.
                    if nullified_info.rewrite_stmt:
                        branch_bit = nullified_info.taken_br
                        x.value = ir.Const(branch_bit, loc=x.loc)
                        # update the specific definition to the new const
                        defns = func_ir._definitions[x.target.name]
                        repl_idx = defns.index(cond)
                        defns[repl_idx] = x.value

    # Check post dominators of dead nodes from in the original CFG for use of
    # vars that are being removed in the dead blocks which might be referred to
    # by phi nodes.
    #
    # Multiple things to fix up:
    #
    # 1. Cases like:
    #
    # A        A
    # |\       |
    # | B  --> B
    # |/       |
    # C        C
    #
    # i.e. the branch is dead but the block is still alive. In this case CFG
    # simplification will fuse A-B-C and any phi in C can be updated as an
    # direct assignment from the last assigned version in the dominators of the
    # fused block.
    #
    # 2. Cases like:
    #
    #   A        A
    #  / \       |
    # B   C  --> B
    #  \ /       |
    #   D        D
    #
    # i.e. the block C is dead. In this case the phis in D need updating to
    # reflect the collapse of the phi condition. This should result in a direct
    # assignment of the surviving version in B to the LHS of the phi in D.

    new_cfg = compute_cfg_from_blocks(func_ir.blocks)
    dead_blocks = new_cfg.dead_nodes()

    # for all phis that are still in live blocks.
    for phi, lbl in phi2lbl.items():
        if lbl in dead_blocks:
            continue
        new_incoming = [x[0] for x in new_cfg.predecessors(lbl)]
        if set(new_incoming) != set(phi.incoming_blocks):
            # Something has changed in the CFG...
            if len(new_incoming) == 1:
                # There's now just one incoming. Replace the PHI node by a
                # direct assignment
                idx = phi.incoming_blocks.index(new_incoming[0])
                phi2asgn[phi].value = phi.incoming_values[idx]
            else:
                # There's more than one incoming still, then look through the
                # incoming and remove dead
                ic_val_tmp = []
                ic_blk_tmp = []
                for ic_val, ic_blk in zip(
                    phi.incoming_values, phi.incoming_blocks
                ):
                    if ic_blk in dead_blocks:
                        continue
                    else:
                        ic_val_tmp.append(ic_val)
                        ic_blk_tmp.append(ic_blk)
                phi.incoming_values.clear()
                phi.incoming_values.extend(ic_val_tmp)
                phi.incoming_blocks.clear()
                phi.incoming_blocks.extend(ic_blk_tmp)

    # Remove dead blocks, this is safe as it relies on the CFG only.
    for dead in dead_blocks:
        del func_ir.blocks[dead]

    # if conditions were nullified then consts were rewritten, update
    if nullified_conditions:
        func_ir._consts = consts.ConstantInference(func_ir)

    if DEBUG > 1:
        print("after".center(80, "-"))
        print(func_ir.dump())
