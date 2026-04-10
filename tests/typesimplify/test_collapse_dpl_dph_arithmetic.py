"""
Tests for _collapse_dpl_dph_arithmetic — 16-bit DPTR construction pattern.

The hardware pattern being recognised:
  MOV A, #lo_const
  ADD A, R_lo          ; sets C (carry)
  MOV DPL, A
  MOV A, #hi_const
  ADDC A, R_hi         ; uses C from above
  MOV DPH, A

After AccumFold + C-kill fix this becomes:
  DPL = lo_const + R_lo        (arithmetic Assign, no C in rhs)
  DPH = (hi_const + R_hi) + C  (top-level + C)

_collapse_dpl_dph_arithmetic collapses these to:
  DPTR = 0x{hi_const:02x}{lo_const:02x} + R_hi_R_lo
"""

import pytest
from pseudo8051.passes.typesimplify._dptr import _collapse_dpl_dph_arithmetic
from pseudo8051.ir.hir import Assign, WhileNode
from pseudo8051.ir.expr import Reg, Const, BinOp, Name, RegGroup


def _a(ea, lhs, rhs):
    return Assign(ea, lhs, rhs)


class TestCollapseDplDphArithmetic:

    # ── Positive cases ────────────────────────────────────────────────────────

    def test_register_pair_with_base(self):
        """DPL = 0x31 + R7; DPH = (0xdc + R6) + C  →  DPTR = 0xdc31 + R6R7  (top-level C)"""
        dpl = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R7")))
        dph = _a(0x11, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Reg("R6")), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 1
        node = result[0]
        assert isinstance(node, Assign)
        assert node.lhs == Reg("DPTR")
        # rhs = BinOp(Const(0xdc31), "+", RegGroup(("R6","R7")))
        assert isinstance(node.rhs, BinOp)
        assert node.rhs.op == "+"
        assert isinstance(node.rhs.lhs, Const)
        assert node.rhs.lhs.value == 0xdc31
        assert node.rhs.rhs == RegGroup(("R6", "R7"))

    def test_register_pair_no_base(self):
        """DPL = 0 + R7; DPH = R6 + C  →  DPTR = R6R7  (zero base elided)"""
        dpl = _a(0x20, Reg("DPL"), BinOp(Const(0), "+", Reg("R7")))
        dph = _a(0x21, Reg("DPH"), BinOp(Reg("R6"), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 1
        node = result[0]
        assert node.lhs == Reg("DPTR")
        assert node.rhs == RegGroup(("R6", "R7"))

    def test_named_byte_fields(self):
        """DPL = 0x31 + offset.lo; DPH = (0xdc + offset.hi) + C  →  DPTR = 0xdc31 + offset"""
        dpl = _a(0x30, Reg("DPL"), BinOp(Const(0x31), "+", Name("offset.lo")))
        dph = _a(0x31, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Name("offset.hi")), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 1
        node = result[0]
        assert node.lhs == Reg("DPTR")
        assert isinstance(node.rhs, BinOp)
        assert node.rhs.lhs.value == 0xdc31
        assert node.rhs.rhs == Name("offset")

    def test_lo_only_base(self):
        """DPL = 0x31 + R7; DPH = R6 + C  →  DPTR = 0x0031 + R6R7"""
        dpl = _a(0x40, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R7")))
        dph = _a(0x41, Reg("DPH"), BinOp(Reg("R6"), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 1
        node = result[0]
        assert node.rhs.lhs.value == 0x0031
        assert node.rhs.rhs == RegGroup(("R6", "R7"))

    def test_hi_only_base(self):
        """DPL = R7 + 0; DPH = (0xdc + R6) + C  →  DPTR = 0xdc00 + R6R7"""
        dpl = _a(0x50, Reg("DPL"), BinOp(Reg("R7"), "+", Const(0)))
        dph = _a(0x51, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Reg("R6")), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 1
        node = result[0]
        assert node.rhs.lhs.value == 0xdc00
        assert node.rhs.rhs == RegGroup(("R6", "R7"))

    def test_nested_carry_addc_form(self):
        """DPL = 0x31 + R7; DPH = 0xdc + (R6 + C)  — nested C from ADDC lift + AccumFold.

        AdDcHandler lifts 'addc A, R6' as A += (R6 + C), so AccumFold produces
        DPH = base + (R6 + C) rather than (base + R6) + C.  Both forms must collapse.
        """
        dpl = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R7")))
        # ADDC form: 0xdc + (R6 + C)  — C nested in the rhs of the outer BinOp
        dph = _a(0x11, Reg("DPH"), BinOp(Const(0xdc), "+", BinOp(Reg("R6"), "+", Reg("C"))))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 1
        node = result[0]
        assert node.lhs == Reg("DPTR")
        assert isinstance(node.rhs, BinOp)
        assert node.rhs.lhs.value == 0xdc31
        assert node.rhs.rhs == RegGroup(("R6", "R7"))

    def test_other_register_pairs(self):
        """R4/R5 and R2/R3 pairs should also be recognised."""
        for rhi, rlo in [("R4", "R5"), ("R2", "R3"), ("R0", "R1")]:
            dpl = _a(0x60, Reg("DPL"), BinOp(Const(0x10), "+", Reg(rlo)))
            dph = _a(0x61, Reg("DPH"), BinOp(BinOp(Const(0x20), "+", Reg(rhi)), "+", Reg("C")))
            result = _collapse_dpl_dph_arithmetic([dpl, dph])
            assert len(result) == 1, f"should collapse {rhi}/{rlo}"
            assert result[0].rhs.rhs == RegGroup((rhi, rlo))

    def test_preserves_surrounding_nodes(self):
        """Nodes before and after the pair are passed through unchanged."""
        pre  = _a(0x00, Reg("R0"), Const(1))
        dpl  = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R7")))
        dph  = _a(0x11, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Reg("R6")), "+", Reg("C")))
        post = _a(0x20, Name("x"), Reg("DPTR"))
        result = _collapse_dpl_dph_arithmetic([pre, dpl, dph, post])
        assert len(result) == 3
        assert result[0] is pre
        assert result[1].lhs == Reg("DPTR")
        assert result[2] is post

    def test_recurses_into_while_body(self):
        """Pattern inside a WhileNode body is also collapsed."""
        dpl = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R7")))
        dph = _a(0x11, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Reg("R6")), "+", Reg("C")))
        loop = WhileNode(0x00, Reg("C"), [dpl, dph])
        result = _collapse_dpl_dph_arithmetic([loop])
        assert len(result) == 1
        assert isinstance(result[0], WhileNode)
        body = result[0].body_nodes
        assert len(body) == 1
        assert body[0].lhs == Reg("DPTR")

    def test_constant_hi_variable_lo(self):
        """DPL = XRAM[DPTR] + 0x39; DPH = 0xDC + C  →  DPTR = 0xDC39 + XRAM[DPTR].

        CLR A + ADDC A, #0xDC pattern: hi byte is a pure constant (no operand),
        lo byte comes from a memory read.  Asymmetric case.
        """
        from pseudo8051.ir.expr import XRAMRef
        dpl = _a(0x10, Reg("DPL"), BinOp(XRAMRef(Reg("DPTR")), "+", Const(0x39)))
        dph = _a(0x11, Reg("DPH"), BinOp(Const(0xdc), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 1
        node = result[0]
        assert node.lhs == Reg("DPTR")
        assert isinstance(node.rhs, BinOp)
        assert node.rhs.op == "+"
        assert node.rhs.lhs.value == 0xdc39
        assert isinstance(node.rhs.rhs, XRAMRef)

    def test_clr_addc_form_before_simplify_arithmetic(self):
        """DPH = 0 + (0xDC + C) — 'CLR A; ADDC A, #0xDC' before +0 is folded.

        AccumFold on 'A=0; A+=(0xDC+C); DPH=A' produces BinOp(0, '+', BinOp(0xDC, '+', C)).
        _split_const_operand must fold the leading 0 so c_hi=0xDC, not c_hi=0.
        """
        from pseudo8051.ir.expr import XRAMRef
        dpl = _a(0x10, Reg("DPL"), BinOp(XRAMRef(Reg("DPTR")), "+", Const(0x39)))
        # Pre-_simplify_arithmetic form: 0 + (0xDC + C)
        dph = _a(0x11, Reg("DPH"),
                 BinOp(Const(0), "+", BinOp(Const(0xdc), "+", Reg("C"))))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 1
        node = result[0]
        assert node.lhs == Reg("DPTR")
        assert node.rhs.lhs.value == 0xdc39

    # ── Negative cases ─────────────────────────────────────────────────────────

    def test_no_carry_in_dph_no_collapse(self):
        """DPH without + C should not be collapsed."""
        dpl = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R7")))
        dph = _a(0x11, Reg("DPH"), BinOp(Const(0xdc), "+", Reg("R6")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 2

    def test_dpl_pure_register_no_collapse(self):
        """DPL = Reg (no arithmetic) should not trigger the pattern."""
        from pseudo8051.ir.expr import Regs
        dpl = _a(0x10, Reg("DPL"), Reg("R7"))  # pure register copy — no BinOp
        dph = _a(0x11, Reg("DPH"), BinOp(Reg("R6"), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 2

    def test_mismatched_pair_no_collapse(self):
        """R6 (hi) and R5 (lo) don't form a standard pair — no collapse."""
        dpl = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R5")))
        dph = _a(0x11, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Reg("R6")), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 2

    def test_odd_hi_register_no_collapse(self):
        """R7 (odd) as hi byte doesn't form a standard pair — no collapse."""
        dpl = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R6")))
        dph = _a(0x11, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Reg("R7")), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 2

    def test_carry_in_dpl_no_collapse(self):
        """DPL containing Reg('C') should not trigger the pattern."""
        dpl = _a(0x10, Reg("DPL"), BinOp(BinOp(Const(0x31), "+", Reg("R7")), "+", Reg("C")))
        dph = _a(0x11, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Reg("R6")), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 2

    def test_mismatched_named_fields_no_collapse(self):
        """Different parent names for .hi/.lo → no collapse."""
        dpl = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Name("foo.lo")))
        dph = _a(0x11, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Name("bar.hi")), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, dph])
        assert len(result) == 2

    def test_dph_not_adjacent_no_collapse(self):
        """An intervening non-DPH node prevents collapse."""
        dpl  = _a(0x10, Reg("DPL"), BinOp(Const(0x31), "+", Reg("R7")))
        mid  = _a(0x15, Reg("R0"), Const(42))
        dph  = _a(0x11, Reg("DPH"), BinOp(BinOp(Const(0xdc), "+", Reg("R6")), "+", Reg("C")))
        result = _collapse_dpl_dph_arithmetic([dpl, mid, dph])
        assert len(result) == 3
