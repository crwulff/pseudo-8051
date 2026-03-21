"""
passes/patterns/sign_bit.py — SignBitTestPattern.

Collapses:
    A = R_hi;          (load high byte of a signed ≥16-bit variable)
    if (ACC.7) { … }   (test the sign bit)

into:
    if (var < 0) { … }
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Assign, IfNode
from pseudo8051.ir.expr import Reg
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base  import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo, _type_bytes, _is_signed


def _is_a_load_reg(node: HIRNode) -> Optional[str]:
    """If node is 'A = Rn;', return Rn; else None."""
    if isinstance(node, Assign):
        if node.lhs == Reg("A") and isinstance(node.rhs, Reg):
            return node.rhs.name
    return None


class SignBitTestPattern(Pattern):
    """Replace 'A = R_hi; if (ACC.7) {…}' with 'if (var < 0) {…}'."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        node = nodes[i]

        reg_name = _is_a_load_reg(node)
        if reg_name is None:
            return None

        info = reg_map.get(reg_name)
        if not (info and info.hi == reg_name
                and _type_bytes(info.type) >= 2 and _is_signed(info.type)):
            return None
        if i + 1 >= len(nodes):
            return None
        nxt = nodes[i + 1]
        if not isinstance(nxt, IfNode):
            return None

        cond = nxt.condition
        cond_str = cond.render() if hasattr(cond, 'render') else str(cond)
        if cond_str != "ACC.7":
            return None

        dbg("typesimp", f"  sign-test: {info.name} < 0")
        replacement = IfNode(
            ea         = nxt.ea,
            condition  = f"{info.name} < 0",
            then_nodes = simplify(nxt.then_nodes, reg_map),
            else_nodes = simplify(nxt.else_nodes, reg_map),
        )
        return ([replacement], i + 2)
