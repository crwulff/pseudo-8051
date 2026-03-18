"""
passes/patterns/xram_group_read.py — XRAMGroupReadPattern.

Collapses a sequential XRAM byte-by-byte read into a typed pointer dereference:

    [DPTR = base_addr;]                     optional DPTR setup
    A = XRAM[base_addr]; Rn0 = A;
    DPTR++;  A = XRAM[base_addr+1]; Rn1 = A;
    DPTR++;  A = XRAM[base_addr+2]; Rn2 = A;
    ...

into a single typed pointer read:

    int32_t b = *(int32_t*)base_addr;

When the read is immediately followed by a statement that references the
register pair, the pointer expression is folded directly into that statement:

    R4R5R6R7 = mul32(R4R5R6R7, R0R1R2R3);  →  a = mul32(a, *(int32_t*)base_addr);
"""

import re
from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Statement
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _fold_into_stmt, _type_bytes,
)

_RE_DPTR_SET   = re.compile(r'^DPTR = (.+?);')
_RE_XRAM_READ  = re.compile(r'^A = XRAM\[(.+)\];$')
_RE_REG_FROM_A = re.compile(r'^(\w+) = A;$')
_RE_DPTR_INC   = re.compile(r'^DPTR\+\+;$')


def _scan_xram_group(nodes: List[HIRNode], start: int,
                     vinfo: VarInfo) -> Optional[Tuple[str, int]]:
    """
    Attempt to match a sequential XRAM read starting at nodes[start].

    Pattern (n = len(vinfo.regs)):
        [DPTR = base_addr;]                    optional
        A = XRAM[base_addr]; Rn0 = A;
        DPTR++;  A = XRAM[next]; Rn1 = A;     repeated for remaining regs

    The registers must be matched in the order given by vinfo.regs (high→low).

    Returns (base_addr_str, end_index) on success, None on failure.
    end_index is the first node after the matched sequence.
    """
    regs = vinfo.regs   # high → low
    n    = len(regs)
    if n < 2:
        return None

    i = start

    # Optionally consume a "DPTR = base_addr;" setup statement
    if i < len(nodes) and isinstance(nodes[i], Statement):
        if _RE_DPTR_SET.match(nodes[i].text):
            i += 1

    base_addr_str: Optional[str] = None

    for k, expected_reg in enumerate(regs):
        if k > 0:
            # Expect DPTR++; before the next XRAM read
            if i >= len(nodes) or not isinstance(nodes[i], Statement):
                return None
            if not _RE_DPTR_INC.match(nodes[i].text):
                return None
            i += 1

        # Expect A = XRAM[addr];
        if i >= len(nodes) or not isinstance(nodes[i], Statement):
            return None
        m_xram = _RE_XRAM_READ.match(nodes[i].text)
        if not m_xram:
            return None
        if k == 0:
            base_addr_str = m_xram.group(1)
        i += 1

        # Expect Rn = A;
        if i >= len(nodes) or not isinstance(nodes[i], Statement):
            return None
        m_reg = _RE_REG_FROM_A.match(nodes[i].text)
        if not m_reg or m_reg.group(1) != expected_reg:
            return None
        i += 1

    if base_addr_str is None:
        return None

    return (base_addr_str, i)


class XRAMGroupReadPattern(Pattern):
    """
    Collapse byte-by-byte XRAM reads into a typed pointer dereference,
    optionally folding the result directly into a following statement.
    """

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        # Try largest register groups first
        candidates = sorted(
            {v for v in reg_map.values() if isinstance(v, VarInfo) and len(v.regs) >= 2},
            key=lambda v: len(v.regs), reverse=True,
        )
        for vinfo in candidates:
            result = _scan_xram_group(nodes, i, vinfo)
            if result is None:
                continue
            base_addr, end_i = result
            ptr_expr = f"*({vinfo.type}*){base_addr}"
            dbg("typesimp", f"  xram-group-read: {vinfo.name} = {ptr_expr}")

            # Try to fold the pointer expression into the immediately following
            # statement when it references the pair name.
            next_node = nodes[end_i] if end_i < len(nodes) else None
            next_text = next_node.text if isinstance(next_node, Statement) else ""
            if next_text:
                folded = _fold_into_stmt(next_text, vinfo.pair_name, ptr_expr, reg_map)
                if folded is None and vinfo.name != vinfo.pair_name:
                    folded = _fold_into_stmt(next_text, vinfo.name, ptr_expr, reg_map)
                if folded is not None:
                    return ([Statement(nodes[i].ea, folded)], end_i + 1)

            # No fold: emit a typed declaration using the logical name
            return ([Statement(nodes[i].ea,
                               f"{vinfo.type} {vinfo.name} = {ptr_expr};")], end_i)

        return None
