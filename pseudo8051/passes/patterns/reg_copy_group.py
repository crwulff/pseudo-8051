"""
passes/patterns/reg_copy_group.py — RegCopyGroupPattern.

Recognises a consecutive sequence of single-register copy statements that
together copy a complete multi-byte variable into a new set of registers.
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Assign
from pseudo8051.ir.expr import Reg, Regs
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo


def _as_reg_copy(node: HIRNode) -> Optional[tuple]:
    """Return (dst_name, src_name) if node is a simple Reg=Reg copy; else None."""
    if isinstance(node, Assign):
        if (isinstance(node.lhs, Regs) and node.lhs.is_single
                and isinstance(node.rhs, Regs) and node.rhs.is_single):
            return (node.lhs.name, node.rhs.name)
    return None


class RegCopyGroupPattern(Pattern):
    """Drop register-copy sequences and propagate the source variable name."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        node = nodes[i]
        pair = _as_reg_copy(node)
        if pair is None:
            return None

        dst0, src0 = pair

        vinfo = reg_map.get(src0)
        if not isinstance(vinfo, VarInfo) or vinfo.xram_sym or len(vinfo.regs) < 2:
            return None
        if vinfo.regs[0] != src0:
            return None

        n = len(vinfo.regs)
        if i + n > len(nodes):
            return None

        dsts: List[str] = []
        for k in range(n):
            nd = nodes[i + k]
            mk = _as_reg_copy(nd)
            if mk is None:
                return None
            dk, sk = mk
            if sk != vinfo.regs[k]:
                return None
            dsts.append(dk)

        new_info = VarInfo(vinfo.name, vinfo.type, tuple(dsts))
        for d in dsts:
            reg_map[d] = new_info
        # Evict source registers so the source TypeGroup is killed from extra_groups
        # and cannot appear as a stale ConstGroupPattern candidate.
        for r in vinfo.regs:
            reg_map.pop(r, None)

        dbg("typesimp",
            f"  reg-copy-group: {vinfo.name} {vinfo.regs!r} → {tuple(dsts)!r}, "
            f"dropped {n} copy statements")
        return ([], i + n)
