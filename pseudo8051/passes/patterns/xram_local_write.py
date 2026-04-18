"""
passes/patterns/xram_local_write.py — XRAMLocalWritePattern.

Collapses a sequential XRAM write to a declared local variable into a single
typed assignment.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign, ExprStmt
from pseudo8051.ir.expr import Expr, Reg, Name, Const, XRAMRef, UnaryOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import CombineTransform, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _type_bytes, _const_str,
)


def _node_is_dptr_set(node: HIRNode, sym: str) -> bool:
    """True if node sets DPTR to the address denoted by sym."""
    if not isinstance(node, Assign) or node.lhs != Reg("DPTR"):
        return False
    rhs = node.rhs
    if isinstance(rhs, Name):
        return rhs.name == sym
    if isinstance(rhs, Const):
        return rhs.alias == sym
    return False


def _node_is_dptr_inc(node: HIRNode) -> bool:
    """True if node is DPTR++."""
    if isinstance(node, ExprStmt):
        if isinstance(node.expr, UnaryOp):
            return node.expr.op == "++" and node.expr.operand == Reg("DPTR")
    return False


def _node_xram_write(node: HIRNode) -> Optional[Tuple[str, Expr]]:
    """If node is XRAM[sym] = expr;, return (sym, rhs_expr).  Otherwise None."""
    if isinstance(node, Assign):
        lhs = node.lhs
        if isinstance(lhs, XRAMRef):
            return (lhs.inner.render(), node.rhs)
    return None


def _scan_xram_local_write(nodes: List[HIRNode], start: int,
                            vinfo: VarInfo) -> Optional[Tuple[List[Expr], int]]:
    """Match a sequential XRAM write to the local described by vinfo."""
    n = _type_bytes(vinfo.type)
    if n < 1:
        return None

    i = start

    # Optionally consume matching DPTR setup
    if i < len(nodes) and _node_is_dptr_set(nodes[i], vinfo.xram_sym):
        i += 1

    byte_exprs: List[Expr] = []

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


def _build_value_str(byte_exprs: List[Expr], type_str: str) -> Optional[Tuple[int, str]]:
    """Return (int_value, display_str) if all byte_exprs are Const, else None."""
    const_vals = [e.value if isinstance(e, Const) else None for e in byte_exprs]
    if any(v is None for v in const_vals):
        return None
    value = 0
    for v in const_vals:
        value = (value << 8) | (v & 0xFF)
    return (value, _const_str(value, type_str))


class XRAMLocalWritePattern(CombineTransform):
    """Collapse byte-by-byte XRAM writes to a declared local into a typed assignment."""

    def produce(self,
               nodes:    List[HIRNode],
               i:        int,
               reg_map:  Dict[str, VarInfo],
               simplify: Simplify) -> Optional[Tuple[HIRNode, int]]:
        candidates = sorted(
            [v for v in reg_map.values() if isinstance(v, VarInfo) and v.xram_sym],
            key=lambda v: _type_bytes(v.type), reverse=True,
        )
        for vinfo in candidates:
            result = _scan_xram_local_write(nodes, i, vinfo)
            if result is None:
                continue
            byte_exprs, end_i = result
            value_result = _build_value_str(byte_exprs, vinfo.type)
            if value_result is not None:
                int_val, value_str = value_result
                rhs_expr: Expr = Const(int_val, alias=value_str)
            elif len(byte_exprs) == 1:
                rhs_expr = byte_exprs[0]
                value_str = rhs_expr.render()
            else:
                continue
            dbg("typesimp", f"  [{hex(nodes[i].ea)}] xram-local-write: {vinfo.name} = {value_str}")
            return (Assign(nodes[i].ea, Name(vinfo.name), rhs_expr), end_i)

        return None
