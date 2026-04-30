"""
passes/patterns/_utils.py — Re-export shim for pattern helper modules.

All symbols previously defined here now live in one of:
  _types.py      — TypeGroup, VarInfo, type helpers, text substitution, const formatting
  _expr_utils.py — Expr tree walking, canonicalization, expression substitution
  _node_utils.py — HIR node traversal, register substitution, folding

This file re-exports everything so existing imports are unaffected.
"""

# Re-export HIR and Expr classes (imported by many pattern files via _utils)
from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, CompoundAssign,  # noqa: F401
                               ExprStmt, ReturnStmt, IfGoto, IfNode, SwitchNode)
from pseudo8051.ir.expr import (  # noqa: F401
    Expr, Reg, Regs, Const, Name, XRAMRef, IRAMRef, RegGroup, ArrayRef, BinOp, UnaryOp,
)

from pseudo8051.passes.patterns._types import (  # noqa: F401
    _parse_array_type, _type_bytes, _is_signed,
    TypeGroup, VarInfo,
    _byte_names,
    _replace_xram_syms, _replace_pairs, _RE_SINGLE_REG, _param_byte_name, _replace_single_regs,
    _parse_int, _const_str,
)

from pseudo8051.passes.patterns._expr_utils import (  # noqa: F401
    _walk_expr, _contains_a,
    _node_a_from_reg, _node_assign_imm, _node_assign_reg,
    _is_reg_free, _regs_in_expr, _fold_unary_const, _canonicalize_expr,
    _subst_pairs_in_expr, _subst_single_regs_in_expr,
    _subst_xram_in_expr, _subst_iram_in_expr, _subst_all_expr,
)

from pseudo8051.passes.patterns._node_utils import (  # noqa: F401
    _apply_expr_subst_to_node, _fold_exprs_in_node,
    _replace_pairs_in_node, _replace_single_regs_in_node,
    _count_reg_uses_in_node, _subst_reg_in_node, _fold_into_node,
)
