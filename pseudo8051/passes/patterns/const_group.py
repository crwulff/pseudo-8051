"""
passes/patterns/const_group.py — ConstGroupPattern.

Collapses a byte-by-byte constant load into a multi-byte register group.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Assign, TypedAssign, ReturnStmt, ExprStmt
from pseudo8051.ir.expr import Reg, Regs, Const, Name, RegGroup
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _replace_pairs, _parse_int, _const_str, _type_bytes,
    _walk_expr, _fold_into_node,
)


def _node_as_assign_imm(node: HIRNode):
    """If node assigns an immediate to a register, return (dst_name, int_value)."""
    if isinstance(node, Assign):
        lhs = node.lhs
        rhs = node.rhs
        if isinstance(lhs, Regs) and lhs.is_single and isinstance(rhs, Const):
            return (lhs.name, rhs.value)
    return None


def _node_as_assign_reg(node: HIRNode):
    """Return (dst_name, src_name) if node is a simple reg=reg assignment."""
    if isinstance(node, Assign):
        lhs = node.lhs
        rhs = node.rhs
        if isinstance(lhs, Regs) and lhs.is_single and isinstance(rhs, Regs) and rhs.is_single:
            return (lhs.name, rhs.name)
    return None


def _scan_const_group(nodes: List[HIRNode], start: int,
                      vinfo: VarInfo) -> Optional[Tuple[int, int]]:
    """
    Scan nodes[start:] for a byte-by-byte constant load into all of vinfo's
    registers.  Returns (combined_value, end_index) on success, None on failure.
    """
    if len(vinfo.regs) < 2:
        return None

    regs_needed = set(vinfo.regs)
    reg_values: Dict[str, int] = {}
    a_value: Optional[int] = None
    i = start
    max_i = min(len(nodes), start + len(regs_needed) * 2 + 2)

    while i < max_i and len(reg_values) < len(regs_needed):
        node = nodes[i]

        imm_result = _node_as_assign_imm(node)
        if imm_result is not None:
            dst, val = imm_result
            if dst == "A":
                a_value = val; i += 1; continue
            if dst in regs_needed and dst not in reg_values:
                reg_values[dst] = val; i += 1; continue
            break

        reg_result = _node_as_assign_reg(node)
        if reg_result is not None:
            dst, src = reg_result
            if dst in regs_needed and src == "A" and a_value is not None \
                    and dst not in reg_values:
                reg_values[dst] = a_value; i += 1; continue
            if dst == "A":
                a_value = None
            break

        break

    if regs_needed != set(reg_values.keys()):
        return None

    value = 0
    for reg in vinfo.regs:   # high → low
        value = (value << 8) | (reg_values[reg] & 0xFF)
    return (value, i)


class ConstGroupPattern(Pattern):
    """
    Collapse a byte-by-byte constant load into a single typed assignment,
    optionally folding the constant directly into a following return statement.
    """

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        candidates = sorted(
            {v for v in reg_map.values()
             if isinstance(v, VarInfo) and len(v.regs) >= 2
             and not v.is_retval_field},
            key=lambda v: len(v.regs), reverse=True,
        )
        if not candidates:
            # No named candidates in reg_map (e.g. all covered by stale retval_field
            # entries that were excluded above).  Peek into the upcoming nodes'
            # call_arg_ann: backward annotation from the following call may have
            # decorated the Rn=A relay nodes (but not the A=const lead-in node that
            # ConstGroupPattern starts from).
            forward_vi: Dict[int, VarInfo] = {}
            for peek in nodes[i : min(i + 20, len(nodes))]:
                ann = getattr(peek, 'ann', None)
                if ann is None:
                    continue
                for vi in ann.call_arg_names.values():
                    if (isinstance(vi, VarInfo) and len(vi.regs) >= 2
                            and not vi.is_retval_field):
                        forward_vi[id(vi)] = vi
            if forward_vi:
                candidates = sorted(forward_vi.values(),
                                    key=lambda v: len(v.regs), reverse=True)
        for vinfo in candidates:
            result = _scan_const_group(nodes, i, vinfo)
            if result is None:
                continue
            value, end_i = result
            const_s = _const_str(value, vinfo.type)
            const_expr = Const(value, alias=const_s)
            dbg("typesimp", f"  const-load: {vinfo.name} = {const_s}")
            # Re-assert vinfo in reg_map so subsequent aliasing uses vinfo.name,
            # not a stale retval name left behind by an earlier RetvalPattern run.
            for r in vinfo.regs:
                reg_map[r] = vinfo

            # Try to fold the constant into the immediately following statement
            next_node = nodes[end_i] if end_i < len(nodes) else None
            if next_node is not None:
                # Try expr-tree fold first: match by register group, then by variable name
                regs_expr = RegGroup(vinfo.regs) if len(vinfo.regs) > 1 else None
                name_expr = Name(vinfo.name) if vinfo.name else None

                folded_node = None
                if regs_expr is not None:
                    folded_node = _fold_into_node(next_node, regs_expr, const_expr, reg_map)
                if folded_node is None and name_expr is not None:
                    folded_node = _fold_into_node(next_node, name_expr, const_expr, reg_map)

                if folded_node is not None:
                    # Span: nodes[i]…next_node; reg_groups from first, call_arg from last
                    from pseudo8051.ir.hir import NodeAnnotation
                    folded_node.ann = NodeAnnotation.merge(nodes[i], next_node)
                    return ([folded_node], end_i + 1)

            # Declare with type
            from pseudo8051.ir.hir import NodeAnnotation
            last = nodes[end_i - 1] if end_i > i else nodes[i]
            out = TypedAssign(nodes[i].ea, vinfo.type,
                              RegGroup(vinfo.regs, alias=vinfo.name), const_expr)
            out.ann = NodeAnnotation.merge(nodes[i], last)
            return ([out], end_i)
        return None
