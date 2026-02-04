# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
This module provides helper functions to find the first line of a function
body.
"""

import ast
import inspect
import textwrap


class FindDefFirstLine(ast.NodeVisitor):
    """
    Attributes
    ----------
    first_stmt_line : int or None
        This stores the first statement line number if the definition is found.
        Or, ``None`` if the definition is not found.
    def_lineno : int or None
        This stores the 'def' line number if the definition is found.
        Or, ``None`` if the definition is not found.
    """

    def __init__(self, name, firstlineno):
        """
        Parameters
        ----------
        name : str
            The function's name (co_name).
        firstlineno : int
            The function's first line number (co_firstlineno), adjusted for
            any offset if the source is a fragment.
        """
        self._co_name = name
        self._co_firstlineno = firstlineno
        self.first_stmt_line = None
        self.def_lineno = None

    def _visit_children(self, node):
        for child in ast.iter_child_nodes(node):
            super().visit(child)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == self._co_name:
            # Name of function matches.

            # The `def` line may match co_firstlineno.
            possible_start_lines = set([node.lineno])
            if node.decorator_list:
                # Has decorators.
                # The first decorator line may match co_firstlineno.
                first_decor = node.decorator_list[0]
                possible_start_lines.add(first_decor.lineno)
            # Does the first lineno match?
            if self._co_firstlineno in possible_start_lines:
                # Yes, we found the function.
                self.def_lineno = node.lineno
                # So, use the first statement line as the first line.
                if node.body:
                    first_stmt = node.body[0]
                    if _is_docstring(first_stmt):
                        # Skip docstring
                        first_stmt = node.body[1]
                    self.first_stmt_line = first_stmt.lineno
                    return
                else:
                    # This is probably unreachable.
                    # Function body cannot be bare. It must at least have
                    # A const string for docstring or a `pass`.
                    pass
        self._visit_children(node)


def _is_docstring(node):
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            return True
    return False


def get_func_body_first_lineno(pyfunc):
    """
    Look up the first line of function body.

    Uses inspect.getsourcelines() which works for both regular .py files
    (via linecache reading from disk) and Jupyter notebook cells (via
    IPython's linecache registration).

    Returns
    -------
    lineno : int; or None
        The first line number of the function body; or ``None`` if the first
        line cannot be determined.
    """
    co = pyfunc.__code__
    try:
        lines, offset = inspect.getsourcelines(pyfunc)
        source = "".join(lines)
        offset = offset - 1
    except (OSError, TypeError):
        return None

    tree = ast.parse(textwrap.dedent(source))
    finder = FindDefFirstLine(co.co_name, co.co_firstlineno - offset)
    finder.visit(tree)
    if finder.first_stmt_line:
        return finder.first_stmt_line + offset
    else:
        # No first line found.
        return None


def get_func_def_lineno(pyfunc):
    """
    Look up the line number of the function definition ('def' line).

    Uses inspect.getsourcelines() which works for both regular .py files
    (via linecache reading from disk) and Jupyter notebook cells (via
    IPython's linecache registration).

    Returns
    -------
    lineno : int; or None
        The line number of the function definition (the 'def' line); or
        ``None`` if it cannot be determined.
    """
    co = pyfunc.__code__
    try:
        lines, offset = inspect.getsourcelines(pyfunc)
        source = "".join(lines)
        offset = offset - 1
    except (OSError, TypeError):
        return None

    tree = ast.parse(textwrap.dedent(source))
    finder = FindDefFirstLine(co.co_name, co.co_firstlineno - offset)
    finder.visit(tree)
    if finder.def_lineno:
        return finder.def_lineno + offset
    else:
        # No def line found.
        return None
