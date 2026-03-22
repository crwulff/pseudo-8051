"""
passes/patterns/accum_fold.py — AccumFoldPattern.

Collapses 8051 A-expression chains such as:

    DPTR = sym;                  # optional DPTR prefix
    A = XRAM[sym];               # or any expr not containing A
    A &= 1;                      # 0 or more compound assigns
    if (A == 0) goto label;      # IfGoto / IfNode / Assign / ReturnStmt terminal

into a single node with A substituted:

    if ((XRAM[sym] & 1) == 0) goto label;

Registered *after* AccumRelayPattern so the pure 2-node relay
(A = expr; target = A; with no ops and no DPTR prefix) is still
owned by AccumRelayPattern.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Assign, CompoundAssign, ReturnStmt, IfGoto, IfNode
from pseudo8051.ir.expr import Expr, Reg, Name, XRAMRef, BinOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo,
    _subst_all_expr,
    _walk_expr,
)

# Map CompoundAssign op to the corresponding binary op
_OP_WITHOUT_EQ = {
    "+=": "+", "-=": "-", "*=": "*", "/=": "/",
    "&=": "&", "|=": "|", "^=": "^",
    "<<=": "<<", ">>=": ">>",
}


def _contains_a(expr: Expr) -> bool:
    """Return True if Reg("A") appears anywhere in the Expr tree."""
    found = [False]

    def _fn(e: Expr) -> Expr:
        if e == Reg("A"):
            found[0] = True
        return e

    _walk_expr(expr, _fn)
    return found[0]


def _subst_a(expr: Expr, replacement: Expr) -> Expr:
    """Replace all occurrences of Reg("A") in expr with replacement."""
    def _fn(e: Expr) -> Expr:
        if e == Reg("A"):
            return replacement
        return e
    return _walk_expr(expr, _fn)


class AccumFoldPattern(Pattern):
    """Collapse A-expression chains into a single terminal node."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        j = i
        dptr_consumed = False

        # ── 1. Optional DPTR prefix ───────────────────────────────────────────
        if j < len(nodes):
            n = nodes[j]
            if (isinstance(n, Assign)
                    and n.lhs == Reg("DPTR")
                    and isinstance(n.rhs, Name)):
                dptr_sym = n.rhs.name
                # Peek at next: must be A = XRAM[dptr_sym]
                if j + 1 < len(nodes):
                    nxt = nodes[j + 1]
                    if (isinstance(nxt, Assign)
                            and nxt.lhs == Reg("A")
                            and isinstance(nxt.rhs, XRAMRef)
                            and isinstance(nxt.rhs.inner, Name)
                            and nxt.rhs.inner.name == dptr_sym):
                        dptr_consumed = True
                        j += 1   # skip DPTR node; j now points at A = XRAM[sym]
                    else:
                        # DPTR node present but next is not the expected XRAM load
                        return None
                else:
                    return None

        # ── 2. A-chain start ──────────────────────────────────────────────────
        if j >= len(nodes):
            return None
        a_start_node = nodes[j]
        if not (isinstance(a_start_node, Assign)
                and a_start_node.lhs == Reg("A")
                and not _contains_a(a_start_node.rhs)):
            return None

        a_expr: Expr = a_start_node.rhs
        j += 1

        # ── 3. Compound assigns ───────────────────────────────────────────────
        num_compound = 0
        while j < len(nodes):
            cn = nodes[j]
            if not (isinstance(cn, CompoundAssign)
                    and cn.lhs == Reg("A")
                    and not _contains_a(cn.rhs)):
                break
            bin_op = _OP_WITHOUT_EQ.get(cn.op)
            if bin_op is None:
                break
            a_expr = BinOp(a_expr, bin_op, cn.rhs)
            num_compound += 1
            j += 1

        # ── 4. Terminal node ──────────────────────────────────────────────────
        if j >= len(nodes):
            return None
        terminal = nodes[j]

        a_expr_subst = _subst_all_expr(a_expr, reg_map)

        # IfGoto: substitute A in condition
        if isinstance(terminal, IfGoto) and _contains_a(terminal.cond):
            new_cond = _subst_a(terminal.cond, a_expr_subst)
            dbg("typesimp", f"  accum_fold (IfGoto): folded {a_expr_subst.render()} into cond")
            return ([IfGoto(a_start_node.ea, new_cond, terminal.label)], j + 1)

        # IfNode: substitute A in condition (Expr or str)
        if isinstance(terminal, IfNode):
            cond = terminal.condition
            if isinstance(cond, Expr) and _contains_a(cond):
                new_cond: object = _subst_a(cond, a_expr_subst)
                new_then = simplify(terminal.then_nodes, reg_map)
                new_else = simplify(terminal.else_nodes, reg_map) if terminal.else_nodes else []
                dbg("typesimp", f"  accum_fold (IfNode expr): folded {a_expr_subst.render()} into cond")
                return ([IfNode(a_start_node.ea, new_cond, new_then, new_else)], j + 1)
            if isinstance(cond, str) and re.search(r'\bA\b', cond):
                rendered = a_expr_subst.render()
                new_cond_str = re.sub(r'\bA\b', rendered, cond)
                new_then = simplify(terminal.then_nodes, reg_map)
                new_else = simplify(terminal.else_nodes, reg_map) if terminal.else_nodes else []
                dbg("typesimp", f"  accum_fold (IfNode str): folded {rendered} into cond")
                return ([IfNode(a_start_node.ea, new_cond_str, new_then, new_else)], j + 1)

        # Assign(target, Reg("A")) where target != A:
        # only fold if there was at least one compound assign or a DPTR prefix
        # (pure 2-node relay without ops is left to AccumRelayPattern).
        if (isinstance(terminal, Assign)
                and terminal.rhs == Reg("A")
                and terminal.lhs != Reg("A")
                and (num_compound > 0 or dptr_consumed)):
            dbg("typesimp", f"  accum_fold (Assign relay): folded {a_expr_subst.render()} into {terminal.lhs.render()}")
            return ([Assign(a_start_node.ea, terminal.lhs, a_expr_subst)], j + 1)

        # ReturnStmt(Reg("A")): only if compound > 0 or DPTR consumed
        if (isinstance(terminal, ReturnStmt)
                and terminal.value == Reg("A")
                and (num_compound > 0 or dptr_consumed)):
            dbg("typesimp", f"  accum_fold (ReturnStmt): folded {a_expr_subst.render()}")
            return ([ReturnStmt(a_start_node.ea, a_expr_subst)], j + 1)

        return None
