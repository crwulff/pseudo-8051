"""
passes/patterns/_utils.py — Shared types and helpers for pattern modules.

VarInfo, type introspection, register-pair text substitution,
and constant formatting are here so each pattern file can import
what it needs without pulling in the full typesimplify module.
"""

import re
import sys
from typing import Dict, List, Optional, Tuple

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
    """One named variable occupying one or more consecutive registers or an XRAM address."""

    def __init__(self, name: str, type_str: str, regs: Tuple[str, ...],
                 xram_sym: str = "", is_byte_field: bool = False,
                 xram_addr: int = 0):
        self.name         = name
        self.type         = type_str
        self.regs         = regs           # high → low order, e.g. ('R6', 'R7'); () for XRAM locals
        self.pair_name    = "".join(regs)  # e.g. 'R6R7'; '' for XRAM locals
        self.xram_sym     = xram_sym       # XRAM base address symbol, e.g. 'EXT_DC8A'; '' for reg vars
        self.is_byte_field = is_byte_field # True for per-byte entries of multi-byte XRAM locals
        self.xram_addr    = xram_addr      # raw integer XRAM address (0 for register vars)

    @property
    def hi(self) -> Optional[str]:
        """Most-significant register, or None for single-byte vars."""
        return self.regs[0] if len(self.regs) >= 2 else None

    @property
    def lo(self) -> Optional[str]:
        """Least-significant register."""
        return self.regs[-1] if self.regs else None


# ── XRAM byte-field helpers ───────────────────────────────────────────────────

def _byte_names(var_name: str, n: int) -> List[str]:
    """Per-byte field names for a multi-byte XRAM local.

    Returns ['var.hi', 'var.lo'] for 2-byte vars, ['var.b0'...'var.bn'] for larger.
    Names are ordered high-byte first (big-endian, matching 8051 XRAM layout).
    """
    if n == 2:
        return [f"{var_name}.hi", f"{var_name}.lo"]
    return [f"{var_name}.b{i}" for i in range(n)]


def _replace_xram_syms(text: str, reg_map: Dict[str, "VarInfo"]) -> str:
    """Replace XRAM[sym] and *sym references with declared XRAM local variable names.

    For multi-byte locals the per-byte names (is_byte_field=True entries) take
    priority over the full-variable name so that individual byte accesses are
    rendered as e.g. ``var1.hi`` / ``var1.lo`` rather than the base name.
    """
    # Build sym → display name, letting byte-field entries override full-var entries.
    sym_map: Dict[str, str] = {}
    for vinfo in reg_map.values():
        if not isinstance(vinfo, VarInfo) or not vinfo.xram_sym:
            continue
        if vinfo.is_byte_field:
            sym_map[vinfo.xram_sym] = vinfo.name          # byte names take priority
        elif vinfo.xram_sym not in sym_map:
            sym_map[vinfo.xram_sym] = vinfo.name          # full-var as fallback

    if not sym_map:
        return text

    def _repl_xram(m: "re.Match") -> str:
        return sym_map.get(m.group(1).strip(), m.group(0))

    def _repl_deref(m: "re.Match") -> str:
        return sym_map.get(m.group(1), m.group(0))

    text = re.sub(r'XRAM\[([^\]]+)\]', _repl_xram, text)
    text = re.sub(r'\*([A-Za-z_]\w*)', _repl_deref, text)
    return text


# ── Register-pair text substitution ──────────────────────────────────────────

def _replace_pairs(text: str, reg_map: Dict[str, VarInfo]) -> str:
    """
    Replace register-pair tokens (e.g. R6R7) with the variable name.
    Longest keys first; word-boundary match.  Single-register names untouched.
    XRAM-local VarInfo entries (xram_sym != '') are skipped — they are handled
    by dedicated patterns so their symbol names are not blindly substituted.
    """
    for key in sorted((k for k in reg_map
                       if len(k) > 2 and isinstance(reg_map[k], VarInfo)),
                      key=len, reverse=True):
        if reg_map[key].xram_sym:
            continue   # XRAM locals: leave substitution to XRAMLocalWritePattern
        text = re.sub(r"\b" + re.escape(key) + r"\b", reg_map[key].name, text)
    return text


# ── Constant formatting ───────────────────────────────────────────────────────

def _parse_int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def _const_str(value: int, type_str: str) -> str:
    _c = sys.modules.get("pseudo8051.constants")
    if _c is None or not getattr(_c, "USE_HEX", True):
        return str(value)
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
