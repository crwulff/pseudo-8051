"""
passes/patterns/neg16.py — Neg16Pattern.

Collapses the 7-statement 8051 16-bit two's-complement negation into:
    var = -var;

Handles both Assign (expression-tree) and legacy Statement nodes.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement, Assign, CompoundAssign
from pseudo8051.ir.expr import Reg, Const, BinOp, UnaryOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo


def _is_clr_zero(node: HIRNode, reg: str) -> bool:
    """True if node clears reg to zero (Assign or Statement form)."""
    if isinstance(node, Assign):
        return node.lhs == Reg(reg) and node.rhs == Const(0)
    if isinstance(node, Statement):
        return re.match(rf'^{re.escape(reg)} = 0;$', node.text) is not None
    return False


def _is_subb_reg(node: HIRNode) -> Optional[str]:
    """If node is 'A -= Rn + C;', return Rn; else None."""
    if isinstance(node, CompoundAssign):
        if (node.lhs == Reg("A") and node.op == "-="
                and isinstance(node.rhs, BinOp)
                and node.rhs.op == "+"
                and node.rhs.rhs == Reg("C")
                and isinstance(node.rhs.lhs, Reg)):
            return node.rhs.lhs.name
    if isinstance(node, Statement):
        m = re.match(r"^A -= (\w+) \+ C;", node.text)
        if m:
            return m.group(1)
    return None


def _is_store_a(node: HIRNode, reg: str) -> bool:
    """True if node is 'reg = A;'."""
    if isinstance(node, Assign):
        return node.lhs == Reg(reg) and node.rhs == Reg("A")
    if isinstance(node, Statement):
        return node.text == f"{reg} = A;"
    return False


class Neg16Pattern(Pattern):
    """Collapse 7-statement SUBB negation into 'var = -var;'."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        if i + 7 > len(nodes):
            return None
        ns = nodes[i:i + 7]

        # Fixed positions: C=0, A=0, ..., A=0, ...
        if not _is_clr_zero(ns[0], "C"):
            return None
        if not _is_clr_zero(ns[1], "A"):
            return None
        if not _is_clr_zero(ns[4], "A"):
            return None

        r_lo = _is_subb_reg(ns[2])
        r_hi = _is_subb_reg(ns[5])
        if r_lo is None or r_hi is None:
            return None

        if not _is_store_a(ns[3], r_lo):
            return None
        if not _is_store_a(ns[6], r_hi):
            return None

        info_lo = reg_map.get(r_lo)
        info_hi = reg_map.get(r_hi)
        if not (info_lo and info_hi and info_lo is info_hi
                and info_lo.lo == r_lo and info_lo.hi == r_hi):
            return None

        dbg("typesimp", f"  neg16: {info_lo.name} = -{info_lo.name}")
        return ([Statement(nodes[i].ea, f"{info_lo.name} = -{info_lo.name};")], i + 7)
