"""
passes/patterns/sign_bit.py — SignBitTestPattern.

Collapses:
    A = R_hi;          (load high byte of a signed ≥16-bit variable)
    if (ACC.7) { … }   (test the sign bit)

into:
    if (var < 0) { … }
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement, IfNode
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base  import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo, _type_bytes, _is_signed


class SignBitTestPattern(Pattern):
    """Replace 'A = R_hi; if (ACC.7) {…}' with 'if (var < 0) {…}'."""

    _RE = re.compile(r"^A = (\w+);$")

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        node = nodes[i]
        if not isinstance(node, Statement):
            return None
        m = self._RE.match(node.text)
        if not m:
            return None
        info = reg_map.get(m.group(1))
        if not (info and info.hi == m.group(1)
                and _type_bytes(info.type) >= 2 and _is_signed(info.type)):
            return None
        if i + 1 >= len(nodes):
            return None
        nxt = nodes[i + 1]
        if not (isinstance(nxt, IfNode) and nxt.condition == "ACC.7"):
            return None
        dbg("typesimp", f"  sign-test: {info.name} < 0")
        replacement = IfNode(
            ea         = nxt.ea,
            condition  = f"{info.name} < 0",
            then_nodes = simplify(nxt.then_nodes, reg_map),
            else_nodes = simplify(nxt.else_nodes, reg_map),
        )
        return ([replacement], i + 2)
