"""
passes/typesimplify/_xram_loads.py — XRAM local load consolidation.

Exports:
  _consolidate_xram_local_loads   merge multi-byte XRAM reads into pair assignments
  _subst_reg_in_scope             forward-substitute a register replacement up to its kill point
"""

import re
from typing import Dict, List

from pseudo8051.ir.hir import (HIRNode, Assign, IfNode, WhileNode, ForNode,
                                DoWhileNode, SwitchNode)
from pseudo8051.ir.expr import Reg, Regs as RegExpr, RegGroup, Name as NameExpr
from pseudo8051.passes.patterns._utils import VarInfo, _type_bytes, _byte_names, _subst_reg_in_node
from pseudo8051.passes.typesimplify._dptr import _is_dptr_inc_node, _as_dph_assign
from pseudo8051.constants import dbg

_RE_BYTE_FIELD = re.compile(r'^(.+)\.(?:hi|lo|b\d+)$')


# ── Register substitution within a scope ─────────────────────────────────────

def _subst_reg_in_scope(nodes: List[HIRNode], reg: str,
                         replacement) -> List[HIRNode]:
    """
    Replace Reg/Name(reg) → replacement in read positions of nodes,
    recursing into structured bodies.  Stops at the first node that writes
    reg (i.e. redefines it), leaving that node and all subsequent nodes
    unchanged.  Used by _consolidate_xram_local_loads to apply single-load
    substitutions immediately within the current scope.
    """
    result = []
    for i, node in enumerate(nodes):
        patched = _subst_reg_in_node(node, reg, replacement)
        if patched is not None:
            result.append(patched)
        else:
            result.append(node.map_bodies(
                lambda ns: _subst_reg_in_scope(ns, reg, replacement)))
        # If this node writes reg on any execution path, subsequent nodes may
        # see a different value — stop.  Use possibly_killed() so that structured
        # nodes (IfNode, WhileNode, etc.) whose bodies sometimes write reg are
        # treated as kill barriers even though their written_regs is frozenset().
        if reg in node.written_regs or reg in node.possibly_killed():
            result.extend(nodes[i + 1:])
            return result
    return result


# ── XRAM local load consolidation ────────────────────────────────────────────

def _consolidate_xram_local_loads(nodes: List[HIRNode],
                                   reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Consolidate post-relay XRAM local byte loads into register-pair assignments.

    Two patterns handled:
    1. Rhi=Name("X.hi"); [DPTR++;...]; Rlo=Name("X.lo"); → RegGroup(Rhi,Rlo)=Name("X")
       Mutates reg_map["RhiRlo"] = VarInfo("X", type, (Rhi, Rlo)).
    2. Rn=Name("local") where "local" is a 1-byte XRAM local (no byte-field suffix)
       Kept as-is; mutates reg_map[Rn] = VarInfo("local", type, (Rn,), is_param=True).
    """
    def _parent_name(s):
        m = _RE_BYTE_FIELD.match(s)
        return m.group(1) if m else None

    def _find_parent_vinfo(parent_nm):
        for v in reg_map.values():
            if (isinstance(v, VarInfo) and v.name == parent_nm
                    and v.xram_sym and not v.is_byte_field):
                return v
        return None

    def _as_byte_assign(n):
        """Return (reg_name, name_str) if n is 'Reg(r) = Name(s)'; else None."""
        if (isinstance(n, Assign)
                and isinstance(n.lhs, RegExpr) and n.lhs.is_single
                and isinstance(n.rhs, NameExpr)):
            return (n.lhs.name, n.rhs.name)
        return None

    nodes = list(nodes)
    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        ba = _as_byte_assign(node)
        if ba is not None:
            reg0, bname0 = ba
            parent_nm = _parent_name(bname0)
            if parent_nm is not None:
                parent_vinfo = _find_parent_vinfo(parent_nm)
                if parent_vinfo is not None:
                    n_bytes = _type_bytes(parent_vinfo.type)
                    expected_bnames = _byte_names(parent_nm, n_bytes, parent_vinfo.type)
                    if bname0 == expected_bnames[0] and n_bytes >= 2:
                        regs = [reg0]
                        j = i + 1
                        for k in range(1, n_bytes):
                            while j < len(nodes) and _is_dptr_inc_node(nodes[j]):
                                j += 1
                            if j >= len(nodes):
                                break
                            next_ba = _as_byte_assign(nodes[j])
                            if next_ba is None or next_ba[1] != expected_bnames[k]:
                                break
                            regs.append(next_ba[0])
                            j += 1
                        if len(regs) == n_bytes:
                            from pseudo8051.ir.hir import NodeAnnotation as _NA
                            new_vinfo = VarInfo(parent_nm, parent_vinfo.type, tuple(regs))
                            for r in regs:
                                reg_map[r] = new_vinfo
                            last_byte_node = nodes[j - 1]
                            merged_ann = _NA.merge(node, last_byte_node)
                            # Check if call_arg_ann provides a register alias (e.g. 'dividend').
                            # active_regs is narrowed per-write-node (e.g. {R7} for the last
                            # byte), so we check full_regs (the full callee param definition)
                            # instead.  Also aggregate from all consumed nodes, not just last.
                            lhs_alias = None
                            regs_fs = frozenset(regs)
                            candidate_groups = list(merged_ann.call_arg_ann) if merged_ann else []
                            for idx in range(i, j - 1):
                                n = nodes[idx]
                                if n.ann and n.ann.call_arg_ann:
                                    candidate_groups.extend(n.ann.call_arg_ann)
                            for g in candidate_groups:
                                if (g.name and not g.xram_sym and g.full_regs
                                        and frozenset(g.full_regs) >= regs_fs):
                                    lhs_alias = g.name
                                    break
                            # If lo byte landed in DPL and next node is DPH = Rhi,
                            # collapse to DPTR = var instead of RegGroup = var.
                            consumed = list(nodes[i:j])
                            if (regs[-1] == "DPL" and j < len(nodes)
                                    and _as_dph_assign(nodes[j]) == regs[0]):
                                new_node = Assign(node.ea, Reg("DPTR"),
                                                  NameExpr(parent_nm))
                                new_node.ann = merged_ann
                                new_node.source_nodes = consumed + [nodes[j]]
                                out.append(new_node)
                                j += 1
                                dbg("typesimp",
                                    f"  [{hex(node.ea)}] xram-pair-consolidate: DPTR = {parent_nm}"
                                    f" (via {regs[0]} + DPH)")
                            else:
                                lhs_expr = RegGroup(tuple(regs), alias=lhs_alias)
                                new_node = Assign(node.ea, lhs_expr, NameExpr(parent_nm))
                                new_node.ann = merged_ann
                                new_node.source_nodes = consumed
                                out.append(new_node)
                                label = lhs_alias or "".join(regs)
                                dbg("typesimp",
                                    f"  [{hex(node.ea)}] xram-pair-consolidate: {label} = {parent_nm}")
                            i = j
                            continue
            else:
                # No byte suffix — check if it's a 1-byte XRAM local
                for v in reg_map.values():
                    if (isinstance(v, VarInfo) and v.name == bname0
                            and v.xram_sym and not v.is_byte_field
                            and _type_bytes(v.type) == 1):
                        new_vinfo = VarInfo(bname0, v.type, (reg0,), is_param=False)
                        reg_map[reg0] = new_vinfo
                        dbg("typesimp", f"  [{hex(node.ea)}] xram-single-load: {reg0} = {bname0}")
                        nodes[i + 1:] = _subst_reg_in_scope(
                            nodes[i + 1:], reg0, NameExpr(bname0))
                        break

        # Recurse into structured nodes with a scope-local copy of reg_map so
        # that arm-internal mutations don't leak into the outer scope.
        out.append(node.map_bodies(
            lambda ns: _consolidate_xram_local_loads(ns, dict(reg_map))))
        i += 1
    return out
