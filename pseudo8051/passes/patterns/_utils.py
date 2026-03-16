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


# ── Statement folding helper ──────────────────────────────────────────────────

def _fold_into_stmt(next_text: str, name_to_fold: str,
                    replacement: str,
                    reg_map: Dict[str, "VarInfo"]) -> Optional[str]:
    """
    Try to substitute `name_to_fold` → `replacement` in `next_text`.

    For assignment statements (`LHS = RHS;`) the substitution is applied only
    to the RHS so the left-hand side is not accidentally replaced.  For return
    statements and standalone expressions the substitution is unrestricted.

    Returns the folded text (with _replace_pairs applied), or None when
    `name_to_fold` does not appear in an expression position.
    """
    if not next_text:
        return None
    pat = re.compile(r'\b' + re.escape(name_to_fold) + r'\b')
    if not pat.search(next_text):
        return None

    if next_text.startswith("return "):
        return _replace_pairs(pat.sub(replacement, next_text), reg_map)

    eq_pos = next_text.find(" = ")
    if eq_pos > 0:
        # Assignment: only replace in the RHS
        rhs = next_text[eq_pos + 3:]
        if pat.search(rhs):
            new_rhs = pat.sub(replacement, rhs)
            return _replace_pairs(next_text[:eq_pos + 3] + new_rhs, reg_map)
        return None  # name_to_fold only in LHS — not a fold opportunity

    # Standalone expression (e.g. a bare function call)
    return _replace_pairs(pat.sub(replacement, next_text), reg_map)
