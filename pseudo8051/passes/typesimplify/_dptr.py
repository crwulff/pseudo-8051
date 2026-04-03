"""
passes/typesimplify/_dptr.py — DPTR-related post-simplify passes.

Exports:
  _is_dptr_inc_node      predicate: ExprStmt(DPTR++)
  _collapse_dpl_dph      merge DPH/DPL register-pair writes into DPTR assignment
  _dptr_live_after       flow-sensitive: is current DPTR value read downstream?
  _prune_orphaned_dptr_inc  remove DPTR++ nodes with no downstream use
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import (HIRNode, Assign, ExprStmt, IfNode, WhileNode,
                                ForNode, DoWhileNode, SwitchNode)
from pseudo8051.ir.expr import (UnaryOp, Reg as RegExpr, RegGroup as RegGroupExpr,
                                 Name as NameExpr)
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.constants import dbg


# ── Shared predicate ──────────────────────────────────────────────────────────

def _is_dptr_inc_node(node: HIRNode) -> bool:
    """True for ExprStmt(DPTR++) — a data-pointer advance node."""
    return (isinstance(node, ExprStmt)
            and isinstance(node.expr, UnaryOp)
            and node.expr.op == "++"
            and node.expr.operand == RegExpr("DPTR"))


# ── DPL/DPH → DPTR collapsing ────────────────────────────────────────────────

def _as_dph_assign(n) -> Optional[str]:
    """Return Rhi name if n is Assign(Reg('DPH'), Reg(Rhi)); else None."""
    if (isinstance(n, Assign)
            and isinstance(n.lhs, RegExpr) and n.lhs.name == "DPH"
            and isinstance(n.rhs, RegExpr)):
        return n.rhs.name
    return None


def _as_dpl_assign(n) -> Optional[str]:
    """Return Rlo name if n is Assign(Reg('DPL'), Reg(Rlo)); else None."""
    if (isinstance(n, Assign)
            and isinstance(n.lhs, RegExpr) and n.lhs.name == "DPL"
            and isinstance(n.rhs, RegExpr)):
        return n.rhs.name
    return None


def _is_call_setup_assign(node: HIRNode) -> bool:
    """True for Assign(Reg/RegGroup, Name/Const) — a consolidated register-setup node."""
    from pseudo8051.ir.expr import Const
    return (isinstance(node, Assign)
            and isinstance(node.lhs, (RegExpr, RegGroupExpr))
            and isinstance(node.rhs, (NameExpr, Const)))


def _collapse_dpl_dph(nodes: List[HIRNode],
                       reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Collapse paired DPH/DPL byte assignments into a single DPTR assignment.

    DPH = Rhi; [skippable...] DPL = Rlo;  →  DPTR = RhiRlo;  (or DPTR = var;)
    DPL = Rlo; [skippable...] DPH = Rhi;  →  same

    Skippable = _is_call_setup_assign or _is_dptr_inc_node.
    """
    recursed = [node.map_bodies(lambda ns: _collapse_dpl_dph(ns, reg_map))
                for node in nodes]

    out: List[HIRNode] = []
    dead: set = set()   # indices consumed as the partner half of a pair

    for i, node in enumerate(recursed):
        if i in dead:
            continue

        rhi = _as_dph_assign(node)   # DPH = Rhi case
        rlo = _as_dpl_assign(node)   # DPL = Rlo case
        if rhi is None and rlo is None:
            out.append(node)
            continue

        # Search forward for the partner, skipping setup/dptr++ nodes.
        partner_idx = None
        partner_val = None
        for k in range(i + 1, len(recursed)):
            if k in dead:
                continue
            if rhi is not None:
                rlo2 = _as_dpl_assign(recursed[k])
                if rlo2 is not None:
                    partner_idx, partner_val = k, rlo2
                    break
            else:
                rhi2 = _as_dph_assign(recursed[k])
                if rhi2 is not None:
                    partner_idx, partner_val = k, rhi2
                    break
            if not (_is_call_setup_assign(recursed[k]) or _is_dptr_inc_node(recursed[k])):
                break   # non-skippable node blocks the search

        if partner_idx is None:
            out.append(node)
            continue

        # Build the collapsed DPTR assignment.
        if rhi is not None:
            reg_hi, reg_lo = rhi, partner_val
        else:
            reg_hi, reg_lo = partner_val, rlo

        pair_key = reg_hi + reg_lo
        vinfo = reg_map.get(pair_key)
        rhs = (NameExpr(vinfo.name)
               if isinstance(vinfo, VarInfo)
               else RegGroupExpr((reg_hi, reg_lo)))
        out.append(Assign(node.ea, RegExpr("DPTR"), rhs))
        dead.add(partner_idx)
        dbg("typesimp", f"  [{hex(node.ea)}] dpl-dph-collapse: DPTR = {pair_key}")

    return out


# ── Flow-sensitive DPTR liveness ──────────────────────────────────────────────

def _dptr_live_after(nodes: List[HIRNode]) -> bool:
    """
    Return True if the current DPTR value is read by any downstream node,
    stopping early when an assignment kills it.

    Flow-sensitive: a DPTR write (e.g. DPTR = _dest) or DPTR++ stops the scan.
    Recurses into IfNode/WhileNode/ForNode bodies conservatively.
    """
    for node in nodes:
        if (isinstance(node, Assign)
                and isinstance(node.lhs, RegExpr) and node.lhs.name == "DPTR"):
            return False  # DPTR overwritten; old value dead
        if _is_dptr_inc_node(node):
            return False  # Another DPTR++ also overwrites the old value
        if isinstance(node, IfNode):
            if (_dptr_live_after(node.then_nodes)
                    or _dptr_live_after(node.else_nodes)):
                return True
        elif isinstance(node, (WhileNode, ForNode)):
            if _dptr_live_after(node.body_nodes):
                return True
        else:
            if "DPTR" in node.name_refs():
                return True
    return False


# ── Orphaned DPTR++ pruning ───────────────────────────────────────────────────

def _prune_orphaned_dptr_inc(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Remove DPTR++ nodes that have no downstream DPTR reference.

    Must run after _propagate_values so that XRAM[DPTR] expressions have
    been resolved to XRAM[name] before we check for surviving references.
    Recurses into bodies first so nested orphans are eliminated before outer
    nodes are evaluated.  Uses _dptr_live_after (flow-sensitive).
    """
    nodes = [node.map_bodies(_prune_orphaned_dptr_inc) for node in nodes]
    result: List[HIRNode] = []
    for i, node in enumerate(nodes):
        if _is_dptr_inc_node(node) and not _dptr_live_after(nodes[i + 1:]):
            dbg("typesimp", f"  [{hex(node.ea)}] prune-orphaned-dptr++")
            continue
        result.append(node)
    return result
