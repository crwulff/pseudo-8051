"""
passes/patterns/const_group.py — ConstGroupPattern.

Collapses a byte-by-byte constant load into a multi-byte register group:
    A = 0x00;  R4 = A;  R5 = A;  R6 = 0x5d;  R7 = 0xc0;

into a single typed assignment:
    int32_t retval = 0x00005dc0;

When the load is immediately followed by a return statement that references
the register pair, the constant is folded directly into that statement:
    return div32(R4R5R6R7, R0R1R2R3);  →  return div32(0x00005dc0, divisor);
"""

import re
from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Statement
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _replace_pairs, _parse_int, _const_str, _type_bytes,
)

_RE_ASSIGN_IMM = re.compile(r"^(\w+) = (0x[0-9a-fA-F]+|\d+);$")
_RE_ASSIGN_REG = re.compile(r"^(\w+) = (\w+);$")


def _scan_const_group(nodes: List[HIRNode], start: int,
                      vinfo: VarInfo) -> Optional[Tuple[int, int]]:
    """
    Scan nodes[start:] for a byte-by-byte constant load into all of vinfo's
    registers.  Understands:
      Rn = literal;   — direct constant load
      A  = literal;   — A used as a zero/constant carrier
      Rn = A;         — forwarding A value to a register

    Returns (combined_value, end_index) on success, None on failure.
    end_index is the index of the first node after the matched sequence.
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
        if not isinstance(node, Statement):
            break

        m_imm = _RE_ASSIGN_IMM.match(node.text)
        if m_imm:
            dst, val = m_imm.group(1), _parse_int(m_imm.group(2))
            if dst == "A":
                a_value = val; i += 1; continue
            if dst in regs_needed and dst not in reg_values:
                reg_values[dst] = val; i += 1; continue
            break   # constant load to an unrelated register — stop

        m_reg = _RE_ASSIGN_REG.match(node.text)
        if m_reg:
            dst, src = m_reg.group(1), m_reg.group(2)
            if dst in regs_needed and src == "A" and a_value is not None \
                    and dst not in reg_values:
                reg_values[dst] = a_value; i += 1; continue
            if dst == "A":
                a_value = None   # A overwritten with unknown value
            break   # unrecognised register move — stop

        break   # any other statement — stop

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
        # Try largest register groups first to avoid a 2-byte match masking a 4-byte one
        candidates = sorted(
            {v for v in reg_map.values() if len(v.regs) >= 2},
            key=lambda v: len(v.regs), reverse=True,
        )
        for vinfo in candidates:
            result = _scan_const_group(nodes, i, vinfo)
            if result is None:
                continue
            value, end_i = result
            const_s = _const_str(value, vinfo.type)
            dbg("typesimp", f"  const-load: {vinfo.name} = {const_s}")

            # Fold into the following return statement when it references the pair.
            # Handles both bare 'return R4R5R6R7;' and tail call forms like
            # 'return func(R4R5R6R7, R0R1R2R3);'.
            next_node = nodes[end_i] if end_i < len(nodes) else None
            next_text = next_node.text if isinstance(next_node, Statement) else ""
            pair_in_next = (vinfo.pair_name in next_text or
                            (vinfo.name != vinfo.pair_name and
                             re.search(r"\b" + re.escape(vinfo.name) + r"\b", next_text)))
            if next_text.startswith("return ") and pair_in_next:
                folded = next_text.replace(vinfo.pair_name, const_s)
                folded = _replace_pairs(folded, reg_map)
                return ([Statement(nodes[i].ea, folded)], end_i + 1)

            # Otherwise declare with type (synthetic 'retval' gets its type shown)
            return ([Statement(nodes[i].ea,
                               f"{vinfo.type} {vinfo.name} = {const_s};")], end_i)
        return None
