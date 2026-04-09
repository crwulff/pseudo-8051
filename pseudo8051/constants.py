"""
constants.py — 8051 / RTD2660 constants and address-resolution helpers.

Migrated from IDA_8051_Pseudocode.py (top-level constants section).
"""

import os as _os

# ── Debug output ──────────────────────────────────────────────────────────────
# Set to True to write pass-by-pass diagnostics to /tmp/<tag>.dbg.
DEBUG = True

_DBG_DIR = "/tmp/pseudo8051"
_dbg_opened: set = set()   # tags whose .dbg file has been opened (write) this session


def reset_debug_session() -> None:
    """Clear the opened-file registry so the next dbg() call per tag overwrites."""
    _os.makedirs(_DBG_DIR, exist_ok=True)
    _dbg_opened.clear()


def dbg(tag: str, msg: str) -> None:
    """Write a debug line to /tmp/<tag>.dbg.

    The first call for each tag in a session opens the file in write mode
    (clearing any previous content); subsequent calls append.
    """
    if not DEBUG:
        return
    _os.makedirs(_DBG_DIR, exist_ok=True)
    path = _os.path.join(_DBG_DIR, f"{tag}.dbg")
    mode = 'w' if tag not in _dbg_opened else 'a'
    _dbg_opened.add(tag)
    with open(path, mode) as f:
        f.write(msg + '\n')

# ── Display options ────────────────────────────────────────────────────────────
# When True, integer constants are rendered in hexadecimal; when False, decimal.
# Toggled at runtime via the right-click menu in the pseudocode viewer.
USE_HEX: bool = True

import ida_name
import ida_segment

# ─────────────────────────────────────────────────────────────────────────────
# IDA 8051 processor-module constants
# ─────────────────────────────────────────────────────────────────────────────

REG_DPTR         = 51   # IDA register number for DPTR
PHRASE_AT_R0     = 0    # @R0  — indirect via R0
PHRASE_AT_R1     = 1    # @R1  — indirect via R1
PHRASE_AT_DPTR   = 2    # @DPTR — used by MOVX
PHRASE_AT_A_DPTR = 3    # @A+DPTR — used by MOVC

# Standard 8051 SFR names (direct-address byte range 0x80–0xFF)
SFR_NAMES = {
    0x80: "P0",
    0x81: "SP",
    0x82: "DPL",
    0x83: "DPH",
    0x87: "PCON",
    0x88: "TCON",
    0x89: "TMOD",
    0x8A: "TL0",
    0x8B: "TL1",
    0x8C: "TH0",
    0x8D: "TH1",
    0x90: "P1",
    0x98: "SCON",
    0x99: "SBUF",
    0xA0: "P2",
    0xA8: "IE",
    0xB0: "P3",
    0xB8: "IP",
    0xD0: "PSW",
    0xE0: "A",
    0xF0: "B",
}

# Registers tracked for liveness / parameter inference, in display order.
PARAM_REGS      = frozenset({"A", "R0", "R1", "R2", "R3", "R4",
                              "R5", "R6", "R7", "B", "DPTR"})
PARAM_REG_ORDER = ["A", "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "B", "DPTR"]


# ─────────────────────────────────────────────────────────────────────────────
# Address / name helpers
# ─────────────────────────────────────────────────────────────────────────────

def lookup_name(ea: int) -> str:
    """Return IDA's symbolic name for ea, or a hex string fallback."""
    name = ida_name.get_name(ea)
    return name if name else hex(ea)


def resolve_ext_addr(dptr_val: int) -> str:
    """
    Map a 16-bit DPTR value to a name in the EXT segment.
    Returns the IDA symbol name if one exists, otherwise hex(dptr_val).
    """
    seg = ida_segment.get_segm_by_name("EXT")
    if seg is None:
        return hex(dptr_val)
    ea = seg.start_ea + (dptr_val & 0xFFFF)
    name = ida_name.get_name(ea)
    return name if name else hex(dptr_val)
