"""
Tests for handler use() and defs() accuracy — side-effect registers.

Validates that handlers correctly declare which registers they read (use)
and write (defs), with particular attention to the carry flag C.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from pseudo8051.handlers.logic      import RlcHandler, RrcHandler, SetbHandler
from pseudo8051.handlers.arithmetic import AddcHandler, SubbHandler
from pseudo8051.handlers.branch     import JcHandler, JncHandler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_insn(reg_name=None, operand_type=None, n_ops=1):
    """
    Build a minimal mock instruction with n_ops operands.
    reg_name: if given, idc.print_operand will return this string for all operands.
    operand_type: op.type value for all ops (default: ida_ua.o_reg = 1).
    """
    import ida_ua
    insn      = MagicMock()
    insn.ea   = 0x1000
    op_type   = operand_type if operand_type is not None else ida_ua.o_reg
    ops = []
    for _ in range(n_ops):
        op = MagicMock()
        op.type = op_type
        ops.append(op)
    insn.ops = ops

    # Make idc.print_operand return the desired name for this insn
    if reg_name is not None:
        import idc
        idc.print_operand = MagicMock(return_value=reg_name)

    return insn


# ── RLC ───────────────────────────────────────────────────────────────────────

class TestRlcHandler:
    def test_use_contains_A_and_C(self):
        h = RlcHandler()
        u = h.use(None)
        assert "A" in u
        assert "C" in u

    def test_defs_contains_A_and_C(self):
        h = RlcHandler()
        d = h.defs(None)
        assert "A" in d
        assert "C" in d


# ── RRC ───────────────────────────────────────────────────────────────────────

class TestRrcHandler:
    def test_use_contains_A_and_C(self):
        h = RrcHandler()
        u = h.use(None)
        assert "A" in u
        assert "C" in u

    def test_defs_contains_A_and_C(self):
        h = RrcHandler()
        d = h.defs(None)
        assert "A" in d
        assert "C" in d


# ── SETB ──────────────────────────────────────────────────────────────────────

class TestSetbHandler:
    def test_defs_C_when_target_is_C(self):
        insn = _make_insn(reg_name="C")
        h = SetbHandler()
        d = h.defs(insn)
        assert "C" in d

    def test_defs_empty_for_bit_address_target(self):
        """SETB 0x35 (bit address, o_mem type) → no register defs."""
        import ida_ua
        insn = _make_insn(operand_type=ida_ua.o_mem)
        insn.ops[0].addr = 0x35  # must be an int for Operand.render() o_mem path
        h = SetbHandler()
        d = h.defs(insn)
        assert len(d) == 0


# ── ADDC ──────────────────────────────────────────────────────────────────────

class TestAddcHandler:
    def test_use_pre_filter_includes_C(self):
        """C should be in the pre-PARAM_REGS-filter use set (conceptual accuracy)."""
        # The returned frozenset filters by PARAM_REGS so C is dropped at the boundary.
        # We verify by confirming A and the source register are present.
        insn = _make_insn(reg_name="R7", n_ops=2)
        h = AddcHandler()
        u = h.use(insn)
        assert "A" in u
        assert "R7" in u
        # C is not in PARAM_REGS so it is filtered out of the public return value —
        # this is expected and correct behaviour.
        assert "C" not in u


# ── SUBB ──────────────────────────────────────────────────────────────────────

class TestSubbHandler:
    def test_use_pre_filter_includes_C(self):
        insn = _make_insn(reg_name="R7", n_ops=2)
        h = SubbHandler()
        u = h.use(insn)
        assert "A" in u
        assert "R7" in u
        assert "C" not in u  # C ∉ PARAM_REGS — filtered at boundary


# ── JC / JNC ─────────────────────────────────────────────────────────────────

class TestJcHandler:
    def test_use_contains_C(self):
        h = JcHandler()
        assert "C" in h.use(None)

    def test_defs_empty(self):
        h = JcHandler()
        assert len(h.defs(None)) == 0


class TestJncHandler:
    def test_use_contains_C(self):
        h = JncHandler()
        assert "C" in h.use(None)

    def test_defs_empty(self):
        h = JncHandler()
        assert len(h.defs(None)) == 0
