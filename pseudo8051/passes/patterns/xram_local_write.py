"""
passes/patterns/xram_local_write.py — XRAMLocalWritePattern.

Collapses a sequential XRAM write to a declared local variable into a single
typed assignment.
"""

import re
from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign, ExprStmt
from pseudo8051.ir.expr import Reg, Name, XRAMRef, UnaryOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _type_bytes, _const_str,
)

_RE_CONST = re.compile(r'^(0x[0-9a-fA-F]+|\d+)$')


def _parse_const(expr: str) -> Optional[int]:
    s = expr.strip()
    m = _RE_CONST.match(s)
    if not m:
        return None
    return int(m.group(1), 16) if m.group(1).startswith("0x") else int(m.group(1))


def _node_is_dptr_set(node: HIRNode, sym: str) -> bool:
    """True if node sets DPTR = sym."""
    if isinstance(node, Assign):
        return node.lhs == Reg("DPTR") and node.rhs == Name(sym)
    return False


def _node_is_dptr_inc(node: HIRNode) -> bool:
    """True if node is DPTR++."""
    if isinstance(node, ExprStmt):
        if isinstance(node.expr, UnaryOp):
            return node.expr.op == "++" and node.expr.operand == Reg("DPTR")
    return False


def _node_xram_write(node: HIRNode) -> Optional[Tuple[str, str]]:
    """If node is XRAM[sym] = expr;, return (sym, expr_str).  Otherwise None."""
    if isinstance(node, Assign):
        lhs = node.lhs
        if isinstance(lhs, XRAMRef):
            return (lhs.inner.render(), node.rhs.render())
    return None


def _scan_xram_local_write(nodes: List[HIRNode], start: int,
                            vinfo: VarInfo) -> Optional[Tuple[List[str], int]]:
    """Match a sequential XRAM write to the local described by vinfo."""
    n = _type_bytes(vinfo.type)
    if n < 1:
        return None

    i = start

    # Optionally consume matching DPTR setup
    if i < len(nodes) and _node_is_dptr_set(nodes[i], vinfo.xram_sym):
        i += 1

    byte_exprs: List[str] = []

    for k in range(n):
        if k > 0:
            if i >= len(nodes) or not _node_is_dptr_inc(nodes[i]):
                return None
            i += 1

        if i >= len(nodes):
            return None
        write = _node_xram_write(nodes[i])
        if not write:
            return None
        if k == 0 and write[0] != vinfo.xram_sym:
            return None
        byte_exprs.append(write[1])
        i += 1

    return (byte_exprs, i)


def _build_value_str(byte_exprs: List[str], type_str: str) -> Optional[str]:
    const_vals = [_parse_const(e) for e in byte_exprs]
    if any(v is None for v in const_vals):
        return None
    value = 0
    for v in const_vals:
        value = (value << 8) | (v & 0xFF)
    return _const_str(value, type_str)


class XRAMLocalWritePattern(Pattern):
    """Collapse byte-by-byte XRAM writes to a declared local into a typed assignment."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        candidates = sorted(
            [v for v in reg_map.values() if isinstance(v, VarInfo) and v.xram_sym],
            key=lambda v: _type_bytes(v.type), reverse=True,
        )
        for vinfo in candidates:
            result = _scan_xram_local_write(nodes, i, vinfo)
            if result is None:
                continue
            byte_exprs, end_i = result
            value_str = _build_value_str(byte_exprs, vinfo.type)
            if value_str is None:
                if len(byte_exprs) == 1:
                    value_str = byte_exprs[0]
                else:
                    continue
            dbg("typesimp", f"  [{hex(nodes[i].ea)}] xram-local-write: {vinfo.name} = {value_str}")
            return ([Assign(nodes[i].ea, Name(vinfo.name), Name(value_str))], end_i)

        return None
