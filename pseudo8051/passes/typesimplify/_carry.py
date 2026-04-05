"""
passes/typesimplify/_carry.py — Carry-flag and comparison simplification passes.

Exports:
  _simplify_carry_comparison   collapse CLR C + SUBB16 sequence in while-loop conditions
  _simplify_cjne_jnc           fold CJNE(nop-goto) + JNC/JC into typed comparisons
  _simplify_orl_zero_check     fold ORL + JZ idiom into multi-byte zero test
"""

import re
from typing import List, Optional

from pseudo8051.ir.hir import (HIRNode, Assign, CompoundAssign, ExprStmt,
                                IfNode, IfGoto, WhileNode, Label)
from pseudo8051.ir.expr import (BinOp, Const, UnaryOp,
                                 Reg, Regs as RegExpr,
                                 Name as NameExpr)
from pseudo8051.constants import dbg

_RE_BYTE_FIELD = re.compile(r'^(.+)\.(?:hi|lo|b\d+)$')


# ── 16-bit SUBB comparison collapsing ────────────────────────────────────────

def _match_subb16(nodes: List[HIRNode], k: int):
    """Try to match CLR C + SUBB lo + MOV A,Rhi + SUBB hi starting at index k.
    Returns (Rlo_sub, Rhi_min, Rhi_sub) or None."""
    if k + 3 >= len(nodes):
        return None
    n0, n1, n2, n3 = nodes[k], nodes[k+1], nodes[k+2], nodes[k+3]
    if not (isinstance(n0, Assign) and n0.lhs == Reg("C")
            and isinstance(n0.rhs, Const) and n0.rhs.value == 0):
        return None
    if not (isinstance(n1, CompoundAssign) and n1.lhs == Reg("A")
            and n1.op == "-=" and isinstance(n1.rhs, BinOp) and n1.rhs.op == "+"
            and isinstance(n1.rhs.lhs, RegExpr) and n1.rhs.lhs.is_single
            and n1.rhs.rhs == Reg("C")):
        return None
    Rlo_sub = n1.rhs.lhs.name
    if not (isinstance(n2, Assign) and n2.lhs == Reg("A")
            and isinstance(n2.rhs, RegExpr) and n2.rhs.is_single):
        return None
    Rhi_min = n2.rhs.name
    if not (isinstance(n3, CompoundAssign) and n3.lhs == Reg("A")
            and n3.op == "-=" and isinstance(n3.rhs, BinOp) and n3.rhs.op == "+"
            and isinstance(n3.rhs.lhs, RegExpr) and n3.rhs.lhs.is_single
            and n3.rhs.rhs == Reg("C")):
        return None
    Rhi_sub = n3.rhs.lhs.name
    return (Rlo_sub, Rhi_min, Rhi_sub)


def _find_reggroup_name(nodes: List[HIRNode], before_idx: int, target_reg: str) -> Optional[str]:
    """Search nodes[:before_idx] backward for Assign(RegGroup, Name) containing target_reg."""
    for node in reversed(nodes[:before_idx]):
        if (isinstance(node, Assign) and isinstance(node.lhs, RegExpr) and not node.lhs.is_single
                and target_reg in node.lhs.regs and isinstance(node.rhs, NameExpr)):
            return node.rhs.name
    return None


def _try_collapse_subb16(cond, body: List[HIRNode]):
    """
    Scan body for the 4-node SUBB16 sequence and resolve names from preceding
    RegGroup assigns. Returns (new_cond, new_body) or None.
    """
    for k in range(len(body)):
        match = _match_subb16(body, k)
        if match is None:
            continue
        Rlo_sub, Rhi_min, Rhi_sub = match
        subtrahend = _find_reggroup_name(body, k, Rhi_sub)
        if subtrahend is None:
            continue
        minuend = _find_reggroup_name(body, k, Rhi_min)
        if minuend is None:
            continue
        new_body = body[:k] + body[k+4:]
        new_cond = BinOp(NameExpr(minuend), "<", NameExpr(subtrahend))
        dbg("typesimp",
            f"  [{hex(body[k].ea)}] subb16-collapse: {minuend} < {subtrahend}"
            f" (via {Rhi_min},{Rlo_sub} - {Rhi_sub},{Rlo_sub})")
        return (new_cond, new_body)
    return None


def _simplify_carry_comparison(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    For each WhileNode with condition Reg("C"), scan body_nodes for the 4-node
    CLR C + SUBB lo + MOV A + SUBB hi sequence. If found and both operand names
    can be resolved, replace the condition with BinOp(Name(minuend), "<", ...).
    Recurses into nested structured nodes.
    """
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, WhileNode):
            new_body = _simplify_carry_comparison(node.body_nodes)
            if node.condition == Reg("C"):
                transformed = _try_collapse_subb16(node.condition, new_body)
                if transformed is not None:
                    new_cond, new_body = transformed
                    result.append(WhileNode(node.ea, new_cond, new_body))
                    continue
            result.append(WhileNode(node.ea, node.condition, new_body))
        else:
            result.append(node.map_bodies(_simplify_carry_comparison))
    return result


# ── CJNE + JNC/JC carry-comparison simplification ────────────────────────────

def _simplify_cjne_jnc(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Collapse the CJNE (nop-goto) + JNC/JC pattern:

      ExprStmt(expr != const)   ← CJNE: sets C = (expr < const); nop-goto removed
      [optional Label nodes]
      IfNode(!C, body)          ← JNC: jumps when C==0 → expr >= const
    →
      IfNode(expr >= const, body)

    Also handles IfGoto variants and JC (Reg("C")).
    """
    def _is_not_c(c) -> bool:
        return (isinstance(c, UnaryOp) and c.op == "!"
                and c.operand == Reg("C"))

    def _is_c(c) -> bool:
        return c == Reg("C")

    work = [n.map_bodies(_simplify_cjne_jnc) for n in nodes]
    result: List[HIRNode] = []
    i = 0
    while i < len(work):
        node = work[i]
        if (isinstance(node, ExprStmt)
                and isinstance(node.expr, BinOp)
                and node.expr.op == "!="
                and isinstance(node.expr.rhs, Const)):
            j = i + 1
            while j < len(work) and isinstance(work[j], Label):
                j += 1
            if j < len(work):
                next_node = work[j]
                cond_expr = node.expr
                new_cond = None

                if isinstance(next_node, IfNode):
                    if _is_not_c(next_node.condition):
                        new_cond = BinOp(cond_expr.lhs, ">=", cond_expr.rhs)
                    elif _is_c(next_node.condition):
                        new_cond = BinOp(cond_expr.lhs, "<", cond_expr.rhs)
                elif isinstance(next_node, IfGoto):
                    if _is_not_c(next_node.cond):
                        new_cond = BinOp(cond_expr.lhs, ">=", cond_expr.rhs)
                    elif _is_c(next_node.cond):
                        new_cond = BinOp(cond_expr.lhs, "<", cond_expr.rhs)

                if new_cond is not None:
                    repl = next_node.replace_condition(new_cond)
                    result.extend(work[i + 1:j])
                    result.append(repl)
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] cjne-jnc: "
                        f"{cond_expr.render()!r} + carry-branch → {new_cond.render()!r}")
                    i = j + 1
                    continue

        result.append(node)
        i += 1
    return result


# ── ORL + JZ zero-check simplification ───────────────────────────────────────

def _simplify_orl_zero_check(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Recognise the 8051 ORL + JZ idiom for 16-bit zero tests.

    Pattern (consecutive):
      CompoundAssign(Reg("A"), "|=", Name(v))   where v has .hi/.lo/.bN suffix
      IfNode or IfGoto with BinOp(anything, "==" | "!=", Const(0)) condition

    Removes the ORL node and replaces the condition with
    BinOp(Name(parent_of_v), op, Const(0)).
    Recurses into WhileNode / IfNode / ForNode bodies first.
    """
    nodes = [n.map_bodies(_simplify_orl_zero_check) for n in nodes]
    result: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if (isinstance(node, CompoundAssign)
                and node.lhs == Reg("A")
                and node.op == "|="
                and isinstance(node.rhs, NameExpr)):
            m = _RE_BYTE_FIELD.match(node.rhs.name)
            if m and i + 1 < len(nodes):
                parent = m.group(1)
                next_node = nodes[i + 1]
                cond = (next_node.condition if isinstance(next_node, IfNode)
                        else next_node.cond if isinstance(next_node, IfGoto)
                        else None)
                if (isinstance(cond, BinOp)
                        and cond.op in ("==", "!=")
                        and isinstance(cond.rhs, Const) and cond.rhs.value == 0):
                    new_cond = BinOp(NameExpr(parent), cond.op, Const(0))
                    result.append(next_node.replace_condition(new_cond))
                    dbg("typesimp",
                        f"  [{hex(node.ea)}] orl-zero-check: A|={node.rhs.name} → {parent} {cond.op} 0")
                    i += 2
                    continue
        result.append(node)
        i += 1
    return result
