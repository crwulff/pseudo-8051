"""
ir/expr/_prec.py — Shared precedence tables and constant-formatting helper.
"""

import sys

# ── Operator precedence (lower number = tighter binding) ─────────────────────
# Unary = 1 (tightest)
# */%   = 2
# +-    = 3
# <<>>  = 4
# <><=>=  = 5
# ==!=  = 6
# &     = 7
# ^     = 8
# |     = 9
# &&    = 10
# ||    = 11

_BIN_PREC = {
    '*': 2, '/': 2, '%': 2,
    '+': 3, '-': 3,
    '<<': 4, '>>': 4,
    '<': 5, '>': 5, '<=': 5, '>=': 5,
    '==': 6, '!=': 6,
    '&': 7,
    '^': 8,
    '|': 9,
    '&&': 10,
    '||': 11,
}
_UNARY_PREC = 1  # tighter than any binary op


def _const_str(value: int) -> str:
    """Format an integer constant matching Operand.render() behaviour."""
    _c = sys.modules.get("pseudo8051.constants")
    use_hex = getattr(_c, "USE_HEX", True) if _c else True
    return str(value) if (not use_hex or value <= 9) else hex(value)
