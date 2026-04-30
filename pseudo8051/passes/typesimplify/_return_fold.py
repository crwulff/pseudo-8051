"""
passes/typesimplify/_return_fold.py — Return-chain folding pass.

Exports:
  _fold_return_chains   fold return-register assignments into ReturnStmt nodes
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import (HIRNode, Assign, TypedAssign, ReturnStmt,
                                IfNode, WhileNode, ForNode, DoWhileNode, SwitchNode,
                                NodeAnnotation)
from pseudo8051.ir.expr import (Reg, Regs as RegExpr, RegGroup as RegGroupExpr,
                                 Name as NameExpr)
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.constants import dbg


def _try_combine_byte_fields(ret_regs: tuple,
                              exprs_by_reg: Dict[str, "object"]) -> Optional[NameExpr]:
    """Try to combine hi/lo Name byte-fields into a single parent Name.

    E.g. ret_regs=('R2','R1'), exprs={'R2': Name('dest.hi'), 'R1': Name('dest.lo')}
    → Name('dest').  Returns None if the pattern does not match.
    """
    suffixes = ("hi", "lo") if len(ret_regs) == 2 else tuple(f"b{k}" for k in range(len(ret_regs)))
    parent: Optional[str] = None
    for r, suf in zip(ret_regs, suffixes):
        expr = exprs_by_reg.get(r)
        if not isinstance(expr, NameExpr):
            return None
        parts = expr.name.rsplit(".", 1)
        if len(parts) != 2 or parts[1] != suf:
            return None
        if parent is None:
            parent = parts[0]
        elif parts[0] != parent:
            return None
    return NameExpr(parent) if parent else None


def _fold_return_chains(hir: List[HIRNode], ret_regs: tuple,
                         reg_map: Optional[Dict] = None) -> List[HIRNode]:
    """Fold return-register assignments into ReturnStmt.

    Single-register return (len(ret_regs) == 1):
        Assign(ret_reg, expr); ReturnStmt(ret_reg | None) → ReturnStmt(expr)

    Multi-register return (len(ret_regs) > 1):
        Assign(RegGroup(ret_regs), expr); ReturnStmt(None) → ReturnStmt(expr)
        ReturnStmt(None): strip trailing ret-reg self-assigns; produce
            ReturnStmt(Name(pair_name)) from reg_map, or ReturnStmt(RegGroup(ret_regs)).
        If ALL ret_regs assigned consecutively just before ReturnStmt:
            try to collapse byte-field pair (Name("x.hi")/Name("x.lo") → Name("x")),
            else fall back to reg_map pair name.
    """
    ret_reg_set = set(ret_regs)

    def _canon_return_expr():
        if len(ret_regs) > 1:
            if reg_map:
                # Look up by first return register (all share the same VarInfo)
                vinfo = reg_map.get(ret_regs[0])
                if (isinstance(vinfo, VarInfo) and vinfo.name
                        and frozenset(vinfo.regs) == frozenset(ret_regs)):
                    return NameExpr(vinfo.name)
            return RegGroupExpr(ret_regs)
        return None

    def _fold(nodes: List[HIRNode]) -> List[HIRNode]:
        out: List[HIRNode] = []
        i = 0
        while i < len(nodes):
            node = nodes[i]

            # ── Multi-reg: RegGroup assignment immediately before ReturnStmt ─────
            if (len(ret_regs) > 1
                    and isinstance(node, Assign)
                    and isinstance(node.lhs, RegExpr) and not node.lhs.is_single
                    and node.lhs.reg_set() == frozenset(ret_regs)
                    and i + 1 < len(nodes)
                    and isinstance(nodes[i + 1], ReturnStmt)
                    and nodes[i + 1].value is None):
                ret = ReturnStmt(node.ea, node.rhs)
                ret.ann = NodeAnnotation.merge(node, nodes[i + 1])
                ret.source_nodes = [node, nodes[i + 1]]
                out.append(ret)
                dbg("typesimp", f"  [{hex(node.ea)}] fold-return (RegGroup): {node.rhs.render()}")
                i += 2
                continue

            # ── Single-reg: only for single-register returns ──────────────────
            if (len(ret_regs) == 1
                    and isinstance(node, Assign)
                    and isinstance(node.lhs, RegExpr) and node.lhs.is_single
                    and node.lhs.name in ret_reg_set
                    and i + 1 < len(nodes)
                    and isinstance(nodes[i + 1], ReturnStmt)
                    and (nodes[i + 1].value is None
                         or nodes[i + 1].value == Reg(node.lhs.name))):
                ret = ReturnStmt(node.ea, node.rhs)
                ret.ann = NodeAnnotation.merge(node, nodes[i + 1])
                ret.source_nodes = [node, nodes[i + 1]]
                out.append(ret)
                dbg("typesimp", f"  [{hex(node.ea)}] fold-return (single): {node.rhs.render()}")
                i += 2
                continue

            # ── Multi-reg ReturnStmt(None): produce canonical pair expression ──
            if (len(ret_regs) > 1
                    and isinstance(node, ReturnStmt)
                    and node.value is None):
                collected: Dict[str, tuple] = {}
                k = len(out) - 1
                while k >= 0:
                    prev = out[k]
                    if (isinstance(prev, Assign)
                            and isinstance(prev.lhs, RegExpr) and prev.lhs.is_single
                            and prev.lhs.name in ret_reg_set
                            and prev.lhs.name not in collected):
                        collected[prev.lhs.name] = (k, prev.rhs)
                        k -= 1
                    else:
                        break

                combined = None
                if len(collected) == len(ret_regs):
                    exprs_by_reg = {r: collected[r][1] for r in ret_regs}
                    combined = (_try_combine_byte_fields(ret_regs, exprs_by_reg)
                                or _canon_return_expr()
                                or RegGroupExpr(ret_regs))
                    start_idx = min(v[0] for v in collected.values())
                    first_consumed = out[start_idx] if start_idx < len(out) else node
                    del out[start_idx:]
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] fold-return (multi-all): {combined.render()}")
                else:
                    while (out
                           and isinstance(out[-1], Assign)
                           and isinstance(out[-1].lhs, RegExpr) and out[-1].lhs.is_single
                           and out[-1].lhs.name in ret_reg_set
                           and out[-1].rhs == out[-1].lhs):
                        out.pop()
                    first_consumed = node
                    combined = _canon_return_expr()
                    if combined is not None:
                        dbg("typesimp",
                            f"  [{hex(node.ea)}] fold-return (multi-canon): {combined.render()}")

                if combined is not None:
                    ret = ReturnStmt(node.ea, combined)
                    ret.ann = NodeAnnotation.merge(first_consumed, node)
                    ret.source_nodes = [first_consumed, node]
                    out.append(ret)
                else:
                    out.append(node)
                i += 1
                continue

            # ── Recurse into structured bodies ────────────────────────────────
            out.append(node.map_bodies(_fold))
            i += 1
        return out

    return _fold(hir)
