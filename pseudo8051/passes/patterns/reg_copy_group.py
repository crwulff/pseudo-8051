"""
passes/patterns/reg_copy_group.py — RegCopyGroupPattern.

Recognises a consecutive sequence of single-register copy statements that
together copy a complete multi-byte variable into a new set of registers:

    R0 = R4;
    R1 = R5;
    R2 = R6;
    R3 = R7;

where R4–R7 all belong to the *same* non-XRAM VarInfo (e.g. ``retval1``).

On match the statements are dropped and ``reg_map`` is updated so that the
destination registers (R0–R3) and their pair key (``R0R1R2R3``) also resolve
to the same variable.  This propagates retval names into the next call's
argument substitution.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo


_RE_COPY = re.compile(r'^(\w+) = (\w+);$')


class RegCopyGroupPattern(Pattern):
    """Drop register-copy sequences and propagate the source variable name."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        node = nodes[i]
        if not isinstance(node, Statement):
            return None
        m = _RE_COPY.match(node.text)
        if not m:
            return None

        dst0, src0 = m.group(1), m.group(2)

        # Source must be the high byte of a known multi-byte non-XRAM variable
        vinfo = reg_map.get(src0)
        if not isinstance(vinfo, VarInfo) or vinfo.xram_sym or len(vinfo.regs) < 2:
            return None
        if vinfo.regs[0] != src0:
            return None  # must start from the high byte

        n = len(vinfo.regs)
        if i + n > len(nodes):
            return None

        # Collect all n copy statements; verify each source matches vinfo.regs[k]
        dsts: List[str] = []
        for k in range(n):
            nd = nodes[i + k]
            if not isinstance(nd, Statement):
                return None
            mk = _RE_COPY.match(nd.text)
            if not mk:
                return None
            dk, sk = mk.group(1), mk.group(2)
            if sk != vinfo.regs[k]:
                return None
            dsts.append(dk)

        # Update reg_map: map destination registers (and their pair) to a new VarInfo
        new_info = VarInfo(vinfo.name, vinfo.type, tuple(dsts))
        pair = "".join(dsts)
        reg_map[pair] = new_info
        for d in dsts:
            reg_map[d] = new_info

        dbg("typesimp",
            f"  reg-copy-group: {vinfo.name} {vinfo.regs!r} → {tuple(dsts)!r}, "
            f"dropped {n} copy statements")
        return ([], i + n)
