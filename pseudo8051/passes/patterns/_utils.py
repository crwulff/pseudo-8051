"""
passes/patterns/_utils.py — Shared types and helpers for pattern modules.

VarInfo, type introspection, register-pair text substitution,
and constant formatting are here so each pattern file can import
what it needs without pulling in the full typesimplify module.
"""

import re
from typing import Dict, Optional, Tuple

from pseudo8051.ir.hir import HIRNode, Statement  # noqa: F401 (re-exported for patterns)


# ── Type helpers ──────────────────────────────────────────────────────────────

def _type_bytes(t: str) -> int:
    if t in ("bool", "uint8_t",  "int8_t",  "char"):  return 1
    if t in ("uint16_t", "int16_t"):                   return 2
    if t in ("uint32_t", "int32_t"):                   return 4
    return 0


def _is_signed(t: str) -> bool:
    return t in ("int8_t", "int16_t", "int32_t")


# ── VarInfo ───────────────────────────────────────────────────────────────────

class VarInfo:
    """One named variable that occupies one or more consecutive registers."""

    def __init__(self, name: str, type_str: str, regs: Tuple[str, ...]):
        self.name      = name
        self.type      = type_str
        self.regs      = regs           # high → low order, e.g. ('R6', 'R7')
        self.pair_name = "".join(regs)  # e.g. 'R6R7'

    @property
    def hi(self) -> Optional[str]:
        """Most-significant register, or None for single-byte vars."""
        return self.regs[0] if len(self.regs) >= 2 else None

    @property
    def lo(self) -> Optional[str]:
        """Least-significant register."""
        return self.regs[-1] if self.regs else None


# ── Register-pair text substitution ──────────────────────────────────────────

def _replace_pairs(text: str, reg_map: Dict[str, VarInfo]) -> str:
    """
    Replace register-pair tokens (e.g. R6R7) with the variable name.
    Longest keys first; word-boundary match.  Single-register names untouched.
    """
    for key in sorted((k for k in reg_map if len(k) > 2), key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(key) + r"\b", reg_map[key].name, text)
    return text


# ── Constant formatting ───────────────────────────────────────────────────────

def _parse_int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def _const_str(value: int, type_str: str) -> str:
    size = _type_bytes(type_str)
    if size >= 4: return f"0x{value:08x}"
    if size == 2: return f"0x{value:04x}"
    return hex(value) if value > 9 else str(value)
