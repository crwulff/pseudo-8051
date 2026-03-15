"""
passes/patterns/neg16.py — Neg16Pattern.

Collapses the 7-statement 8051 16-bit two's-complement negation:
    C = 0;
    A = 0;  A -= R_lo + C;  R_lo = A;
    A = 0;  A -= R_hi + C;  R_hi = A;

into:
    var = -var;
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo


class Neg16Pattern(Pattern):
    """Collapse 7-statement SUBB negation into 'var = -var;'."""

    _RE_SUBB = re.compile(r"^A -= (\w+) \+ C;")
    _RE_STOR = re.compile(r"^(\w+) = A;$")
    _FIXED   = {0: r"^C = 0;$", 1: r"^A = 0;$", 4: r"^A = 0;$"}

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        if i + 7 > len(nodes):
            return None
        ns = nodes[i:i + 7]

        for idx, pat in self._FIXED.items():
            if not (isinstance(ns[idx], Statement) and re.match(pat, ns[idx].text)):
                return None

        m2 = isinstance(ns[2], Statement) and self._RE_SUBB.match(ns[2].text)
        m3 = isinstance(ns[3], Statement) and self._RE_STOR.match(ns[3].text)
        m5 = isinstance(ns[5], Statement) and self._RE_SUBB.match(ns[5].text)
        m6 = isinstance(ns[6], Statement) and self._RE_STOR.match(ns[6].text)
        if not (m2 and m3 and m5 and m6):
            return None

        r_lo, r_hi = m2.group(1), m5.group(1)
        if m3.group(1) != r_lo or m6.group(1) != r_hi:
            return None

        info_lo = reg_map.get(r_lo)
        info_hi = reg_map.get(r_hi)
        if not (info_lo and info_hi and info_lo is info_hi
                and info_lo.lo == r_lo and info_lo.hi == r_hi):
            return None

        dbg("typesimp", f"  neg16: {info_lo.name} = -{info_lo.name}")
        return ([Statement(nodes[i].ea, f"{info_lo.name} = -{info_lo.name};")], i + 7)
