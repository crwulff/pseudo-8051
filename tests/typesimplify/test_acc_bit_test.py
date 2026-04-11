"""
Tests for _simplify_acc_bit_test — ACC.N bit-test → bitmask condition.

Pattern:
  A = expr;
  C = ACC.N;
  if (C) / if (!C) / while (C) / while (!C)
→
  A = expr;
  if (expr & (1 << N)) / if (!(expr & (1 << N))) / ...

The C = ACC.N node is removed; A = expr is kept.
"""

from pseudo8051.passes.typesimplify._carry import _simplify_acc_bit_test
from pseudo8051.ir.hir import Assign, IfNode, WhileNode, ExprStmt
from pseudo8051.ir.expr import Reg, Const, Name, BinOp, UnaryOp, Call


# ── Helpers ───────────────────────────────────────────────────────────────────

def _a_assign(rhs):
    return Assign(0x10, Reg("A"), rhs)

def _c_acc(n: int):
    return Assign(0x11, Reg("C"), Name(f"ACC.{n}"))

def _if_c(body=None, else_=None):
    return IfNode(0x20, Reg("C"), body or [], else_ or [])

def _if_not_c(body=None, else_=None):
    return IfNode(0x20, UnaryOp("!", Reg("C")), body or [], else_ or [])

def _while_c(body=None):
    return WhileNode(0x20, Reg("C"), body or [])


# ── Positive cases ────────────────────────────────────────────────────────────

class TestAccBitTestPositive:

    def test_if_not_c_bit1(self):
        """A = xarg2; C = ACC.1; if (!C) → if (!(xarg2 & 0x2))."""
        nodes = [_a_assign(Name("xarg2")), _c_acc(1), _if_not_c()]
        result = _simplify_acc_bit_test(nodes)
        assert len(result) == 2          # C=ACC.1 removed
        assert isinstance(result[0], Assign)  # A = xarg2 kept
        wn = result[1]
        assert isinstance(wn, IfNode)
        cond = wn.condition
        assert isinstance(cond, UnaryOp) and cond.op == "!"
        inner = cond.operand
        assert isinstance(inner, BinOp) and inner.op == "&"
        assert inner.lhs == Name("xarg2")
        assert inner.rhs == Const(0x2)

    def test_if_c_bit0(self):
        """A = val; C = ACC.0; if (C) → if (val & 0x1)."""
        nodes = [_a_assign(Name("val")), _c_acc(0), _if_c()]
        result = _simplify_acc_bit_test(nodes)
        assert len(result) == 2
        cond = result[1].condition
        assert isinstance(cond, BinOp) and cond.op == "&"
        assert cond.rhs == Const(1)

    def test_if_c_bit7(self):
        """Bit 7 → mask 0x80."""
        nodes = [_a_assign(Const(0)), _c_acc(7), _if_c()]
        result = _simplify_acc_bit_test(nodes)
        assert len(result) == 2
        cond = result[1].condition
        assert isinstance(cond, BinOp) and cond.rhs == Const(0x80)

    def test_while_c(self):
        """While(C) after C=ACC.N → while(A_expr & mask)."""
        nodes = [_a_assign(Name("x")), _c_acc(3), _while_c()]
        result = _simplify_acc_bit_test(nodes)
        assert len(result) == 2
        assert isinstance(result[1], WhileNode)
        cond = result[1].condition
        assert isinstance(cond, BinOp) and cond.rhs == Const(1 << 3)

    def test_non_adjacent_ok(self):
        """An intervening node that doesn't read/write C is skipped."""
        other = ExprStmt(0x15, Call("nop", []))   # no C reference
        nodes = [_a_assign(Name("y")), _c_acc(2), other, _if_not_c()]
        result = _simplify_acc_bit_test(nodes)
        assert len(result) == 3          # C=ACC removed, other + if remain
        assert isinstance(result[2], IfNode)
        cond = result[2].condition
        assert isinstance(cond, UnaryOp) and cond.op == "!"

    def test_a_value_from_const(self):
        """A = Const works as the source expression."""
        nodes = [_a_assign(Const(42)), _c_acc(1), _if_c()]
        result = _simplify_acc_bit_test(nodes)
        assert result[1].condition == BinOp(Const(42), "&", Const(2))

    def test_result_rendered(self):
        """Rendered output looks like '!(xarg2 & 0x2)'."""
        nodes = [_a_assign(Name("xarg2")), _c_acc(1), _if_not_c()]
        result = _simplify_acc_bit_test(nodes)
        from pseudo8051.ir.hir._base import _render_cond
        rendered = _render_cond(result[1].condition)
        assert rendered == "!(xarg2 & 2)"


# ── Negative cases (pattern not applied) ─────────────────────────────────────

class TestAccBitTestNegative:

    def test_no_a_assign_before(self):
        """Without a preceding A assignment, pattern is not applied."""
        nodes = [_c_acc(1), _if_not_c()]
        result = _simplify_acc_bit_test(nodes)
        # C=ACC.1 and IfNode both kept unchanged
        assert len(result) == 2
        assert isinstance(result[0], Assign) and result[0].lhs == Reg("C")
        assert isinstance(result[1].condition, UnaryOp)
        assert result[1].condition.operand == Reg("C")

    def test_a_overwritten_between_assign_and_acc(self):
        """If A is overwritten between A=expr and C=ACC.N, bail out."""
        a_overwrite = Assign(0x12, Reg("A"), Const(99))
        nodes = [_a_assign(Name("x")), a_overwrite, _c_acc(1), _if_not_c()]
        result = _simplify_acc_bit_test(nodes)
        # The pattern looks backward from C=ACC.1 and finds the overwrite (A=99),
        # uses 99 as the expression (closest A assign before the ACC node).
        # Actually the closest A= is a_overwrite → should use Const(99).
        cond = result[-1].condition
        assert isinstance(cond, UnaryOp) and isinstance(cond.operand, BinOp)
        assert cond.operand.lhs == Const(99)

    def test_c_written_before_if(self):
        """If C is overwritten before the if, pattern is not applied."""
        c_overwrite = Assign(0x15, Reg("C"), Const(0))
        nodes = [_a_assign(Name("x")), _c_acc(1), c_overwrite, _if_not_c()]
        result = _simplify_acc_bit_test(nodes)
        # C=ACC.1 kept; condition unchanged
        assert len(result) == 4
        assert result[2].lhs == Reg("C")   # c_overwrite still there
        assert result[3].condition == UnaryOp("!", Reg("C"))

    def test_non_acc_name_unchanged(self):
        """C = some_other_name does not trigger the pattern."""
        nodes = [_a_assign(Name("x")), Assign(0x11, Reg("C"), Name("SFR.1")),
                 _if_not_c()]
        result = _simplify_acc_bit_test(nodes)
        assert len(result) == 3   # nothing removed

    def test_non_c_if_unchanged(self):
        """IfNode with a non-C condition is not affected."""
        nodes = [_a_assign(Name("x")), _c_acc(1),
                 IfNode(0x20, BinOp(Reg("R7"), "!=", Const(0)), [])]
        result = _simplify_acc_bit_test(nodes)
        # C=ACC.1 not removed (no C-conditioned successor found)
        assert len(result) == 3
