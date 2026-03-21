"""
tests/conftest.py — IDA module stubs installed before any pseudo8051 import.

Provides MagicMock replacements for all IDA Pro modules so the test suite
runs without IDA installed.  Specific integer constants required by the
production code are patched to their real values.
"""

import os
import sys
from unittest.mock import MagicMock

# ── IDA module mocks ──────────────────────────────────────────────────────────

_IDA_MOCKS = [
    'ida_ua', 'idc', 'ida_name', 'ida_funcs', 'ida_gdl',
    'ida_bytes', 'ida_segment', 'ida_nalt', 'ida_typeinf',
    'ida_netnode', 'ida_kernwin', 'ida_idp',
]
for _mod in _IDA_MOCKS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Operand type constants used in operand.py / instruction.py
_ua = sys.modules['ida_ua']
_ua.o_void   = 0
_ua.o_reg    = 1
_ua.o_mem    = 2
_ua.o_phrase = 3
_ua.o_imm    = 5
_ua.o_near   = 6
_ua.o_far    = 7

# BADADDR sentinel used throughout (must be an int for == comparisons)
sys.modules['idc'].BADADDR = 0xFFFFFFFF

# Make _proto_from_ida immediately bail out for all functions:
# get_name_ea returns BADADDR → early return None.
sys.modules['ida_name'].get_name_ea = MagicMock(return_value=0xFFFFFFFF)

# ── sys.path: IDAScripts/ must be importable as the root of pseudo8051 ───────

_here        = os.path.dirname(os.path.abspath(__file__))   # .../IDAScripts/tests
_ida_scripts = os.path.dirname(_here)                       # .../IDAScripts

if _ida_scripts not in sys.path:
    sys.path.insert(0, _ida_scripts)
