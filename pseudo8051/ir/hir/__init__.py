"""
ir/hir/__init__.py — Re-exports all HIR node classes and shared helpers.

All existing `from pseudo8051.ir.hir import ...` statements continue to work.
"""

from pseudo8051.ir.hir._base import (          # noqa: F401
    NodeAnnotation, HIRNode,
    _render_expr, _expr_lines, _ann_field,
    _lhs_written_regs, _Cond, _render_cond,
)
from pseudo8051.ir.hir.assign         import Assign, TypedAssign          # noqa: F401
from pseudo8051.ir.hir.compound_assign import CompoundAssign              # noqa: F401
from pseudo8051.ir.hir.expr_stmt      import ExprStmt                     # noqa: F401
from pseudo8051.ir.hir.return_stmt    import ReturnStmt                   # noqa: F401
from pseudo8051.ir.hir.if_goto        import IfGoto                       # noqa: F401
from pseudo8051.ir.hir.statement      import Statement                    # noqa: F401
from pseudo8051.ir.hir.goto_statement import GotoStatement                # noqa: F401
from pseudo8051.ir.hir.break_stmt     import BreakStmt                    # noqa: F401
from pseudo8051.ir.hir.var_decl       import VarDecl                      # noqa: F401
from pseudo8051.ir.hir.computed_jump  import ComputedJump                 # noqa: F401
from pseudo8051.ir.hir.label          import Label                        # noqa: F401
from pseudo8051.ir.hir.if_node        import IfNode                       # noqa: F401
from pseudo8051.ir.hir.while_node     import WhileNode                    # noqa: F401
from pseudo8051.ir.hir.for_node       import ForNode                      # noqa: F401
from pseudo8051.ir.hir.do_while_node  import DoWhileNode                  # noqa: F401
from pseudo8051.ir.hir.switch_node    import SwitchNode                   # noqa: F401
