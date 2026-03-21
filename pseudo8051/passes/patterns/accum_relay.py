"""
passes/patterns/accum_relay.py — AccumRelayPattern.

Collapses the 8051 idiom of routing a value through the accumulator:

    A = <expr>;
    <target> = A;

into a single statement:

    <target> = <expr>;

Handles both expression-tree nodes (Assign) and legacy Statement nodes.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement, Assign
from pseudo8051.ir.expr import Reg, Expr
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo,
    _replace_xram_syms, _replace_pairs, _replace_single_regs,
    _subst_all_expr,
)

_RE_A_LOAD  = re.compile(r'^A = (.+);$')
_RE_A_STORE = re.compile(r'^(.+) = A;$')


def _subst_all_text(text: str, reg_map: Dict[str, VarInfo]) -> str:
    text = _replace_xram_syms(text, reg_map)
    text = _replace_pairs(text, reg_map)
    text = _replace_single_regs(text, reg_map)
    return text


class AccumRelayPattern(Pattern):
    """Collapse 'A = expr; target = A;' into 'target = expr;'."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        if i + 1 >= len(nodes):
            return None

        n0, n1 = nodes[i], nodes[i + 1]

        # ── Expr-tree path ────────────────────────────────────────────────
        if isinstance(n0, Assign) and isinstance(n1, Assign):
            if (n0.lhs == Reg("A")
                    and n1.rhs == Reg("A")
                    and n1.lhs != Reg("A")
                    and n0.rhs != Reg("A")):
                new_rhs = _subst_all_expr(n0.rhs, reg_map)
                new_lhs = n1.lhs
                dbg("typesimp", f"  accum_relay (expr): {n0.lhs.render()} = {n0.rhs.render()} + {n1.lhs.render()} = {n1.rhs.render()}")
                return ([Assign(n0.ea, new_lhs, new_rhs)], i + 2)

        # ── Legacy Statement path ─────────────────────────────────────────
        if not isinstance(n0, Statement) or not isinstance(n1, Statement):
            return None

        m_load  = _RE_A_LOAD.match(n0.text)
        m_store = _RE_A_STORE.match(n1.text)
        if not m_load or not m_store:
            return None

        expr   = m_load.group(1)
        target = m_store.group(1)

        if expr == "A" or target == "A":
            return None

        new_text = _subst_all_text(f"{target} = {expr};", reg_map)
        dbg("typesimp", f"  accum_relay: {n0.text!r} + {n1.text!r} → {new_text!r}")
        return ([Statement(n0.ea, new_text)], i + 2)
