"""
passes/patterns/xram_local_write.py — XRAMLocalWritePattern.

Collapses a sequential XRAM write to a declared local variable into a single
typed assignment:

    [DPTR = EXT_DCXX;]
    XRAM[EXT_DCXX] = byte_hi_expr;
    DPTR++;
    XRAM[EXT_DCYY] = byte_lo_expr;
    ...

into:

    count = 0x0000;    (for int16_t 'count' at EXT_DCXX when both bytes are 0)

Local variables are declared in the LOCALS table in prototypes.py.

Single-byte locals are also handled (no DPTR++ needed):

    XRAM[EXT_DCXX] = expr;  →  count = expr;
"""

import re
from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Statement
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _type_bytes, _const_str,
)

_RE_DPTR_SET   = re.compile(r'^DPTR = (.+?);')
_RE_XRAM_WRITE = re.compile(r'^XRAM\[(.+)\] = (.+);$')
_RE_DPTR_INC   = re.compile(r'^DPTR\+\+;$')
_RE_CONST      = re.compile(r'^(0x[0-9a-fA-F]+|\d+)$')


def _parse_const(expr: str) -> Optional[int]:
    """Return integer value of a constant expression string, or None."""
    s = expr.strip()
    m = _RE_CONST.match(s)
    if not m:
        return None
    return int(m.group(1), 16) if m.group(1).startswith("0x") else int(m.group(1))


def _scan_xram_local_write(nodes: List[HIRNode], start: int,
                            vinfo: VarInfo) -> Optional[Tuple[List[str], int]]:
    """
    Match a sequential XRAM write to the local described by vinfo.

    Pattern (n = _type_bytes(vinfo.type) bytes):
        [DPTR = xram_sym;]                    optional DPTR setup
        XRAM[xram_sym]   = byte0_expr;
        DPTR++;  XRAM[...] = byte1_expr;      repeated for remaining bytes

    The first write address must equal vinfo.xram_sym; subsequent addresses
    follow via DPTR++ so they are trusted to be sequential.

    Returns (byte_exprs_list, end_index) or None.
    byte_exprs_list is high-byte first (big-endian).
    """
    n = _type_bytes(vinfo.type)
    if n < 1:
        return None

    i = start

    # Optionally consume matching DPTR setup
    if i < len(nodes) and isinstance(nodes[i], Statement):
        m = _RE_DPTR_SET.match(nodes[i].text)
        if m and m.group(1) == vinfo.xram_sym:
            i += 1

    byte_exprs: List[str] = []

    for k in range(n):
        if k > 0:
            # Expect DPTR++ between bytes
            if i >= len(nodes) or not isinstance(nodes[i], Statement):
                return None
            if not _RE_DPTR_INC.match(nodes[i].text):
                return None
            i += 1

        # Expect XRAM[addr] = expr;
        if i >= len(nodes) or not isinstance(nodes[i], Statement):
            return None
        m_w = _RE_XRAM_WRITE.match(nodes[i].text)
        if not m_w:
            return None
        if k == 0 and m_w.group(1) != vinfo.xram_sym:
            return None   # first byte must be at the declared symbol
        byte_exprs.append(m_w.group(2).strip())
        i += 1

    return (byte_exprs, i)


def _build_value_str(byte_exprs: List[str], type_str: str) -> Optional[str]:
    """
    Combine per-byte expressions into a typed value string.

    Currently handles the all-constant case.  Returns None when any byte is
    a non-constant expression (register name, etc.) so those cases fall through
    to the default emitter.
    """
    const_vals = [_parse_const(e) for e in byte_exprs]
    if any(v is None for v in const_vals):
        return None   # mixed / register sources — not yet handled

    value = 0
    for v in const_vals:          # high byte first → big-endian
        value = (value << 8) | (v & 0xFF)
    return _const_str(value, type_str)


class XRAMLocalWritePattern(Pattern):
    """
    Collapse byte-by-byte XRAM writes to a declared local variable into a
    single typed assignment.  Local variables are declared in LOCALS in
    prototypes.py.
    """

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        # Collect XRAM-local VarInfo entries (sorted largest type first)
        candidates = sorted(
            [v for v in reg_map.values() if v.xram_sym],
            key=lambda v: _type_bytes(v.type), reverse=True,
        )
        for vinfo in candidates:
            result = _scan_xram_local_write(nodes, i, vinfo)
            if result is None:
                continue
            byte_exprs, end_i = result
            value_str = _build_value_str(byte_exprs, vinfo.type)
            if value_str is None:
                # Not all constants: emit best-effort raw assignment
                if len(byte_exprs) == 1:
                    value_str = byte_exprs[0]
                else:
                    continue   # multi-byte non-constant — leave for now
            dbg("typesimp", f"  xram-local-write: {vinfo.name} = {value_str}")
            return ([Statement(nodes[i].ea, f"{vinfo.name} = {value_str};")], end_i)

        return None
