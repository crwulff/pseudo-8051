"""
passes/patterns/xram_group_read.py — XRAMGroupReadPattern.

Collapses a sequential XRAM byte-by-byte read into a typed pointer dereference.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign, TypedAssign, ExprStmt
from pseudo8051.ir.expr import Reg, Regs, RegGroup, Name, XRAMRef, UnaryOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import CombineTransform, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _fold_into_node, _type_bytes,
)


def _node_is_dptr_set(node: HIRNode) -> bool:
    if isinstance(node, Assign):
        return node.lhs == Reg("DPTR")
    return False


def _node_is_dptr_inc(node: HIRNode) -> bool:
    if isinstance(node, ExprStmt):
        if isinstance(node.expr, UnaryOp):
            return node.expr.op == "++" and node.expr.operand == Reg("DPTR")
    return False


def _node_xram_read(node: HIRNode) -> Optional[str]:
    """If node is 'A = XRAM[addr];', return addr_str; else None."""
    if isinstance(node, Assign):
        if node.lhs == Reg("A") and isinstance(node.rhs, XRAMRef):
            return node.rhs.inner.render()
    return None


def _node_reg_from_a(node: HIRNode) -> Optional[str]:
    """If node is 'Rn = A;', return Rn; else None."""
    if isinstance(node, Assign):
        if isinstance(node.lhs, Regs) and node.lhs.is_single and node.rhs == Reg("A"):
            return node.lhs.name
    return None


def _scan_xram_group(nodes: List[HIRNode], start: int,
                     vinfo: VarInfo) -> Optional[Tuple[str, int]]:
    regs = vinfo.regs
    n    = len(regs)
    if n < 2:
        return None

    i = start

    # Optionally consume a "DPTR = base_addr;" setup statement
    if i < len(nodes) and _node_is_dptr_set(nodes[i]):
        i += 1

    base_addr_str: Optional[str] = None

    for k, expected_reg in enumerate(regs):
        if k > 0:
            if i >= len(nodes) or not _node_is_dptr_inc(nodes[i]):
                return None
            i += 1

        if i >= len(nodes):
            return None
        addr = _node_xram_read(nodes[i])
        if addr is None:
            return None
        if k == 0:
            base_addr_str = addr
        i += 1

        if i >= len(nodes):
            return None
        reg = _node_reg_from_a(nodes[i])
        if reg is None or reg != expected_reg:
            return None
        i += 1

    if base_addr_str is None:
        return None

    return (base_addr_str, i)


class XRAMGroupReadPattern(CombineTransform):
    """Collapse byte-by-byte XRAM reads into a typed pointer dereference."""

    def produce(self,
               nodes:    List[HIRNode],
               i:        int,
               reg_map:  Dict[str, VarInfo],
               simplify: Simplify) -> Optional[Tuple[HIRNode, int]]:
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
            # Re-assert vinfo in reg_map so subsequent aliasing uses vinfo.name.
            for r in vinfo.regs:
                reg_map[r] = vinfo

            from pseudo8051.ir.hir import NodeAnnotation

            next_node = nodes[end_i] if end_i < len(nodes) else None
            if next_node is not None:
                regs_expr     = RegGroup(vinfo.regs) if len(vinfo.regs) > 1 else Name(vinfo.regs[0])
                var_name_expr = Name(vinfo.name)
                ptr_name_expr = Name(ptr_expr)

                folded_node = _fold_into_node(next_node, regs_expr, ptr_name_expr, reg_map)
                if folded_node is None and vinfo.name:
                    folded_node = _fold_into_node(next_node, var_name_expr, ptr_name_expr, reg_map)

                if folded_node is not None:
                    folded_node.ann = NodeAnnotation.merge(nodes[i], next_node)
                    return (folded_node, end_i + 1)

            last = nodes[end_i - 1] if end_i > i else nodes[i]
            out = TypedAssign(nodes[i].ea, vinfo.type,
                              RegGroup(vinfo.regs, alias=vinfo.name), Name(ptr_expr))
            out.ann = NodeAnnotation.merge(nodes[i], last)
            return (out, end_i)

        return None
