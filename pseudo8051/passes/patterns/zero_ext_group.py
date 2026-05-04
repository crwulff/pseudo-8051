"""
passes/patterns/zero_ext_group.py — ZeroExtGroupPattern.

Collapses a 16-bit register pair where the high byte is set to zero and
the low byte is set to a non-constant expression into a zero-extension cast.

    Rhi = 0            (high byte = constant zero)
    Rlo = non_const    (low byte = any non-constant expression)

into:

    var = (uint16_t)non_const

This only fires for known unsigned 16-bit register pairs; the all-constant
case is handled by ConstGroupPattern.  Skippable inter-pair nodes (DPTR++
increments, residual ExprStmt(Const(...))) are silently skipped.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign, TypedAssign, ExprStmt
from pseudo8051.ir.expr import Regs, RegGroup, Const, Cast, UnaryOp, Reg
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import CombineTransform, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _type_bytes, _is_signed, _fold_into_node,
)


def _is_skippable(node: HIRNode) -> bool:
    """True if node can be skipped between byte-field pair assignments."""
    if isinstance(node, ExprStmt):
        if isinstance(node.expr, UnaryOp):
            return node.expr.op == "++" and node.expr.operand == Reg("DPTR")
        if isinstance(node.expr, Const):
            return True
    return False


def _scan_zero_ext(
    nodes: List[HIRNode],
    start: int,
    hi_reg: str,
    lo_reg: str,
) -> Optional[Tuple[object, int]]:
    """
    Scan nodes[start:] for:
        hi_reg = Const(0)       (immediately or after skippable nodes)
        [skippable nodes]
        lo_reg = non_const_expr

    Returns (lo_expr, end_index) on success, None on failure.
    The caller is responsible for checking that lo_expr is not a Const.
    """
    i = start

    # Must find hi_reg = Const(0)
    if i >= len(nodes):
        return None
    n0 = nodes[i]
    if not (isinstance(n0, Assign)
            and isinstance(n0.lhs, Regs) and n0.lhs.is_single
            and n0.lhs.name == hi_reg
            and isinstance(n0.rhs, Const) and n0.rhs.value == 0):
        return None
    i += 1

    # Skip inter-pair skippable nodes
    while i < len(nodes) and _is_skippable(nodes[i]):
        i += 1

    # Must find lo_reg = <any expression>
    if i >= len(nodes):
        return None
    n1 = nodes[i]
    if not (isinstance(n1, Assign)
            and isinstance(n1.lhs, Regs) and n1.lhs.is_single
            and n1.lhs.name == lo_reg):
        return None

    lo_expr = n1.rhs
    return (lo_expr, i + 1)


class ZeroExtGroupPattern(CombineTransform):
    """
    Collapse a 16-bit pair load where the high byte is zero into a zero-extension cast.

    For a known unsigned 16-bit register pair var (type uint16_t, regs=[Rhi, Rlo]):

        Rhi = 0
        Rlo = expr      (non-constant)

    Produces:

        var = (uint16_t)expr
    """

    def produce(self,
                nodes:    List[HIRNode],
                i:        int,
                reg_map:  Dict[str, VarInfo],
                simplify: Simplify) -> Optional[Tuple[HIRNode, int]]:

        candidates = sorted(
            {v for v in reg_map.values()
             if isinstance(v, VarInfo) and len(v.regs) == 2
             and "." not in v.name
             and _type_bytes(v.type) == 2 and not _is_signed(v.type)},
            key=lambda v: v.name,
        )
        if not candidates:
            return None

        node0 = nodes[i] if i < len(nodes) else None
        dbg("typesimp",
            f"  zero-ext @ {hex(node0.ea) if node0 else '?'}: "
            f"candidates={[v.name for v in candidates]}")

        for vinfo in candidates:
            hi_reg, lo_reg = vinfo.regs  # high byte first, low byte second

            result = _scan_zero_ext(nodes, i, hi_reg, lo_reg)
            if result is None:
                dbg("typesimp", f"  zero-ext: {vinfo.name} scan failed")
                continue

            lo_expr, end_i = result

            # All-constant case is handled by ConstGroupPattern; skip it here
            if isinstance(lo_expr, Const):
                dbg("typesimp", f"  zero-ext: {vinfo.name} lo is const, skip")
                continue

            dbg("typesimp",
                f"  zero-ext: {vinfo.name} = (uint16_t){lo_expr.render()}")

            # Re-assert vinfo in reg_map so subsequent aliasing uses vinfo.name
            for r in vinfo.regs:
                reg_map[r] = vinfo

            cast_expr = Cast(vinfo.type, lo_expr)

            from pseudo8051.ir.hir import NodeAnnotation
            last = nodes[end_i - 1] if end_i > i else nodes[i]

            # Try to fold the cast directly into the immediately following node
            next_node = nodes[end_i] if end_i < len(nodes) else None
            if next_node is not None:
                regs_expr = RegGroup(vinfo.regs)
                from pseudo8051.ir.expr import Name
                name_expr = Name(vinfo.name) if vinfo.name else None

                folded = _fold_into_node(next_node, regs_expr, cast_expr, reg_map)
                if folded is None and name_expr is not None:
                    folded = _fold_into_node(next_node, name_expr, cast_expr, reg_map)

                if folded is not None:
                    folded.ann = NodeAnnotation.merge(nodes[i], next_node)
                    return (folded, end_i + 1)

            out = TypedAssign(nodes[i].ea, vinfo.type,
                              RegGroup(vinfo.regs, alias=vinfo.name), cast_expr)
            out.ann = NodeAnnotation.merge(nodes[i], last)
            return (out, end_i)

        return None
