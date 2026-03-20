"""
passes/patterns/accum_relay.py — AccumRelayPattern.

Collapses the 8051 idiom of routing a value through the accumulator:

    A = <expr>;
    <target> = A;

into a single statement:

    <target> = <expr>;

with full register substitution applied (XRAM symbols, reg pairs, single-reg
params).  This eliminates the A-as-copy-register noise that appears when the
compiler moves a register-resident parameter into an XRAM location:

    A = R7;
    XRAM[DPCD_ADDR_PORT_H] = A;

→  XRAM[DPCD_ADDR_PORT_H] = H;   (when R7 is param H)

The pattern only fires when:
  • node[i]   is "A = <expr>;"   and expr is not just "A"
  • node[i+1] is "<target> = A;" and target is not "A"
  • A does not appear on the left-hand side of node[i+1] (i.e. it is purely
    a read operand in the second statement)
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo,
    _replace_xram_syms, _replace_pairs, _replace_single_regs,
)

_RE_A_LOAD  = re.compile(r'^A = (.+);$')
_RE_A_STORE = re.compile(r'^(.+) = A;$')


def _subst_all(text: str, reg_map: Dict[str, VarInfo]) -> str:
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
        if not isinstance(n0, Statement) or not isinstance(n1, Statement):
            return None

        m_load  = _RE_A_LOAD.match(n0.text)
        m_store = _RE_A_STORE.match(n1.text)
        if not m_load or not m_store:
            return None

        expr   = m_load.group(1)
        target = m_store.group(1)

        # Don't collapse trivial A = A or target = A where target is also A
        if expr == "A" or target == "A":
            return None

        new_text = _subst_all(f"{target} = {expr};", reg_map)
        dbg("typesimp", f"  accum_relay: {n0.text!r} + {n1.text!r} → {new_text!r}")
        return ([Statement(n0.ea, new_text)], i + 2)
