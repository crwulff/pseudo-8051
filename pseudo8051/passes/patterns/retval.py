"""
passes/patterns/retval.py — RetvalPattern.

Renames the return value of a function call to a fresh ``retvalN`` variable:

    R4R5R6R7 = code_7_div32_R4R5R6R7_R0R1R2R3_0(R4R5R6R7, R0R1R2R3);

becomes:

    uint32_t retval1 = code_7_div32_R4R5R6R7_R0R1R2R3_0(dividend, R0R1R2R3);

and ``reg_map`` is updated so subsequent statements using the return
registers resolve to ``retval1``.

The per-function counter is stored in ``reg_map["__n__"]`` as a ``[int]``
list so patterns can increment it without replacing the entry.
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import VarInfo, _replace_pairs, _replace_xram_syms


_RE_CALL_STMT = re.compile(r'^(\w+) = ([A-Za-z_]\w*)\((.*)\);$', re.DOTALL)


class RetvalPattern(Pattern):
    """Rename call return registers to retvalN and emit a typed declaration."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        node = nodes[i]
        if not isinstance(node, Statement):
            return None
        m = _RE_CALL_STMT.match(node.text)
        if not m:
            return None

        lhs, callee, args_raw = m.group(1), m.group(2), m.group(3)

        # LHS must be a known non-XRAM register VarInfo.
        # Direct key lookup works when lhs is a raw register name (e.g. R4R5R6R7).
        # Fallback by name handles the case where a preceding fold pattern already
        # applied _replace_pairs, turning R4R5R6R7 into the variable name (e.g. dividend).
        vinfo = reg_map.get(lhs)
        if not isinstance(vinfo, VarInfo):
            vinfo = next((v for v in reg_map.values()
                          if isinstance(v, VarInfo) and v.name == lhs and v.regs), None)
        if vinfo is None or vinfo.xram_sym or not vinfo.regs:
            return None

        # Callee must have a known prototype with a non-void return type
        from pseudo8051.prototypes import get_proto
        proto = get_proto(callee)
        if proto is None or proto.return_type in ("void", "") or not proto.return_regs:
            return None

        # Allocate a fresh retval name
        counter = reg_map.get("__n__")
        if not isinstance(counter, list):
            return None
        n = counter[0]
        counter[0] += 1
        retval_name = f"retval{n + 1}"

        # Substitute args text
        subst_args = _replace_pairs(_replace_xram_syms(args_raw, reg_map), reg_map)

        # Build new VarInfo for the return registers
        new_info = VarInfo(retval_name, proto.return_type, proto.return_regs)
        pair = "".join(proto.return_regs)
        reg_map[pair] = new_info
        for r in proto.return_regs:
            reg_map[r] = new_info
        # Also update the old VarInfo's pair key (e.g. 'R4R5R6R7') when it
        # differs from the callee's return-register pair (handles non-standard
        # conventions) and when lhs was a raw register key.
        if vinfo.pair_name and vinfo.pair_name != pair:
            reg_map[vinfo.pair_name] = new_info
        if lhs in reg_map and lhs != pair:
            reg_map[lhs] = new_info

        text = f"{proto.return_type} {retval_name} = {callee}({subst_args});"
        dbg("typesimp", f"  retval: {text}")
        return ([Statement(node.ea, text)], i + 1)
