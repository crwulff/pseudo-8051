"""
tests/patterns/test_mul16.py — Mul16Pattern unit tests.

Verifies recognition of the 8051 16×16→16 multiply idiom.
"""

import pytest

from pseudo8051.ir.hir import Assign, CompoundAssign, ExprStmt, NodeAnnotation
from pseudo8051.ir.expr import Reg, Regs, Const, BinOp, Call, RegGroup, Cast, XRAMRef, Name
from pseudo8051.passes.patterns.mul16 import (
    Mul16Pattern, _is_adjacent_hi_lo, _pair2_expr, _find_pair_load_source,
)
from pseudo8051.passes.patterns._utils import VarInfo


def _make_swap(reg: str) -> ExprStmt:
    return ExprStmt(0, Call("swap", [Reg("A"), Reg(reg)]))


def _make_mul_ab(ea: int = 0) -> Assign:
    return Assign(ea, RegGroup(("B", "A"), brace=True), BinOp(Reg("A"), "*", Reg("B")))


def _match(nodes, reg_map=None):
    return Mul16Pattern().match(nodes, 0, reg_map or {}, lambda ns, rm: ns)


def _make_mul16_nodes(Rlo1="R7", Rhi1="R6", Rtemp="R0",
                      lo2=None, hi2=None, base_ea=0x1000):
    """Build the canonical 12-node 16×16 multiply sequence."""
    lo2 = lo2 or Reg("R5")
    hi2 = hi2 or Reg("R4")
    ea = base_ea
    return [
        Assign(ea + 0x00, Reg("A"), Reg(Rlo1)),           # 1: A = Rlo1
        Assign(ea + 0x01, Reg("B"), lo2),                  # 2: B = lo2
        _make_mul_ab(ea + 0x02),                            # 3: {B,A} = A*B
        Assign(ea + 0x03, Reg(Rtemp), Reg("B")),           # 4: Rtemp = B
        _make_swap(Rlo1),                                   # 5: swap(A, Rlo1)
        Assign(ea + 0x05, Reg("B"), hi2),                  # 6: B = hi2
        _make_mul_ab(ea + 0x06),                            # 7: {B,A} = A*B
        CompoundAssign(ea + 0x07, Reg("A"), "+=", Reg(Rtemp)),  # 8: A += Rtemp
        _make_swap(Rhi1),                                   # 9: swap(A, Rhi1)
        Assign(ea + 0x09, Reg("B"), lo2),                  # 10: B = lo2
        _make_mul_ab(ea + 0x0a),                            # 11: {B,A} = A*B
        CompoundAssign(ea + 0x0b, Reg("A"), "+=", Reg(Rhi1)),   # 12: A += Rhi1
    ]


class TestIsAdjacentHiLo:
    def test_r6r7(self):
        assert _is_adjacent_hi_lo("R6", "R7") is True

    def test_r4r5(self):
        assert _is_adjacent_hi_lo("R4", "R5") is True

    def test_r0r1(self):
        assert _is_adjacent_hi_lo("R0", "R1") is True

    def test_r2r3(self):
        assert _is_adjacent_hi_lo("R2", "R3") is True

    def test_odd_hi_rejected(self):
        assert _is_adjacent_hi_lo("R1", "R2") is False

    def test_non_adjacent(self):
        assert _is_adjacent_hi_lo("R6", "R5") is False

    def test_same(self):
        assert _is_adjacent_hi_lo("R6", "R6") is False

    def test_non_registers(self):
        assert _is_adjacent_hi_lo("A", "B") is False


class TestMul16PatternBasic:
    def test_full_sequence_no_reg_map(self):
        """12-node canonical sequence matches without reg_map."""
        nodes = _make_mul16_nodes()
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 12
        assert len(replacement) == 1
        node = replacement[0]
        assert isinstance(node, Assign)
        # LHS is {A, R7} — A is not a natural pair reg so brace=True
        assert isinstance(node.lhs, Regs) and not node.lhs.is_single
        assert set(node.lhs.names) == {"A", "R7"}
        assert node.lhs.brace is True  # A is not hi of natural pair
        # RHS is R6R7 * R4R5 (without reg_map, no alias)
        assert isinstance(node.rhs, BinOp)
        assert node.rhs.op == "*"

    def test_different_pair_r4r5(self):
        """Works with Rhi1=R4, Rlo1=R5 as the first operand."""
        nodes = _make_mul16_nodes(Rlo1="R5", Rhi1="R4", Rtemp="R0",
                                  lo2=Reg("R3"), hi2=Reg("R2"))
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 12
        node = replacement[0]
        assert isinstance(node, Assign)
        assert set(node.lhs.names) == {"A", "R5"}

    def test_consumes_exactly_12_nodes(self):
        """Trailing nodes are not consumed when no Rhi1=A follows."""
        trailing = Assign(0x9000, Reg("R5"), Reg("A"))  # R5≠Rhi1(R6) → not consumed
        nodes = _make_mul16_nodes() + [trailing]
        result = _match(nodes)
        assert result is not None
        _, new_i = result
        assert new_i == 12

    def test_optional_rhi1_equals_a_consumed(self):
        """Optional 13th node Rhi1=A is consumed; LHS becomes R6R7 (brace=False)."""
        save_hi = Assign(0x100c, Reg("R6"), Reg("A"))
        nodes = _make_mul16_nodes() + [save_hi]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 13
        node = replacement[0]
        assert isinstance(node, Assign)
        assert set(node.lhs.names) == {"R6", "R7"}
        assert node.lhs.brace is False  # natural pair — no braces

    def test_wrong_reg_rhi1_equals_a_not_consumed(self):
        """Rhi1=A where the register is not Rhi1 is NOT consumed."""
        save_wrong = Assign(0x100c, Reg("R4"), Reg("A"))
        nodes = _make_mul16_nodes() + [save_wrong]
        result = _match(nodes)
        assert result is not None
        _, new_i = result
        assert new_i == 12

    def test_too_short_returns_none(self):
        """Fewer than 12 nodes → no match."""
        nodes = _make_mul16_nodes()[:11]
        assert _match(nodes) is None

    def test_wrong_rtemp_in_step8(self):
        """Step 8 using wrong Rtemp → no match."""
        nodes = _make_mul16_nodes()
        # Replace step 8 A += R0 with A += R1
        nodes[7] = CompoundAssign(0x1007, Reg("A"), "+=", Reg("R1"))
        assert _match(nodes) is None

    def test_wrong_rhi1_in_step12(self):
        """Step 12 using wrong register → no match."""
        nodes = _make_mul16_nodes()
        nodes[11] = CompoundAssign(0x100b, Reg("A"), "+=", Reg("R7"))
        assert _match(nodes) is None

    def test_lo2_mismatch_returns_none(self):
        """Step 10 lo2 different from step 2 → no match."""
        nodes = _make_mul16_nodes()
        # Replace step 10 B = R5 with B = R3
        nodes[9] = Assign(0x1009, Reg("B"), Reg("R3"))
        assert _match(nodes) is None

    def test_odd_rhi1_rejected(self):
        """Non-standard pair (R1R2 is not even-hi) → no match."""
        nodes = _make_mul16_nodes(Rlo1="R2", Rhi1="R1")
        assert _match(nodes) is None


class TestMul16PatternWithRegMap:
    def _osdaddr_count_regmap(self):
        vi_osd   = VarInfo("osdAddr", "int16_t", ("R6", "R7"))
        vi_count = VarInfo("count",   "int16_t", ("R4", "R5"))
        return {
            "R6": vi_osd,   "R7": vi_osd,   "R6R7": vi_osd,
            "R4": vi_count, "R5": vi_count,  "R4R5": vi_count,
        }

    def test_pair_names_substituted(self):
        """With osdAddr/count reg_map, RHS renders as 'osdAddr * count'."""
        nodes = _make_mul16_nodes(lo2=Reg("R5"), hi2=Reg("R4"))
        reg_map = self._osdaddr_count_regmap()
        result = _match(nodes, reg_map)
        assert result is not None
        node = result[0][0]
        rhs = node.rhs
        assert isinstance(rhs, BinOp) and rhs.op == "*"
        # Both sides should render with variable names
        lhs_str = rhs.lhs.render()
        rhs_str = rhs.rhs.render()
        assert lhs_str == "osdAddr"
        assert rhs_str == "count"

    def test_hi2_const_zero_uses_lo2_only(self):
        """When hi2 = Const(0), pair2 is just lo2 (8-bit operand)."""
        nodes = _make_mul16_nodes(lo2=Reg("R5"), hi2=Const(0))
        vi_count = VarInfo("count", "int16_t", ("R4", "R5"))
        reg_map = {
            "R6": VarInfo("osdAddr", "int16_t", ("R6", "R7")),
            "R7": VarInfo("osdAddr", "int16_t", ("R6", "R7")),
            "R6R7": VarInfo("osdAddr", "int16_t", ("R6", "R7")),
            "R4": vi_count, "R5": vi_count, "R4R5": vi_count,
        }
        result = _match(nodes, reg_map)
        assert result is not None
        node = result[0][0]
        # pair2 should use the known pair (lo2=R5 is part of count pair)
        rhs_str = node.rhs.rhs.render()
        assert rhs_str == "count"

    def test_const_lo2_no_reg_map(self):
        """Constant lo2 (e.g. Const(9)) with hi2=Const(0) → pair2=(uint16_t)9."""
        nodes = _make_mul16_nodes(lo2=Const(9), hi2=Const(0))
        result = _match(nodes)
        assert result is not None
        node = result[0][0]
        rhs_str = node.rhs.rhs.render()
        assert rhs_str == "(uint16_t)9"


class TestPair2Expr:
    def _count_regmap(self):
        vi = VarInfo("count", "int16_t", ("R4", "R5"))
        return {"R4": vi, "R5": vi, "R4R5": vi}

    def test_adjacent_pair_registers(self):
        """R5/R4 with matching VarInfo → aliased to 'count'."""
        reg_map = self._count_regmap()
        result = _pair2_expr(Reg("R5"), Reg("R4"), reg_map)
        assert result.render() == "count"

    def test_lo_only_with_zero_hi(self):
        """R5 is lo of count pair; hi2=Const(0) → still renders as 'count'."""
        reg_map = self._count_regmap()
        result = _pair2_expr(Reg("R5"), Const(0), reg_map)
        assert result.render() == "count"

    def test_const_lo_zero_hi(self):
        """Const(9) with Const(0) hi → zero-extension cast."""
        result = _pair2_expr(Const(9), Const(0), {})
        assert result.render() == "(uint16_t)9"

    def test_general_byte_shift(self):
        """Non-paired, non-zero hi → byte-shift construct."""
        result = _pair2_expr(Reg("R3"), Reg("R2"), {})
        rendered = result.render()
        assert "<<" in rendered
        assert "|" in rendered


class TestFindPairLoadSource:
    def _load_node(self, rhi, rlo, src_expr):
        return Assign(0x1000, RegGroup((rhi, rlo)), src_expr)

    def test_immediately_preceding(self):
        """Load at i-1 → source returned."""
        from pseudo8051.ir.expr import Name
        load = self._load_node("R6", "R7", Name("osdAddr"))
        mul_nodes = _make_mul16_nodes()
        nodes = [load] + mul_nodes
        src = _find_pair_load_source(nodes, 1, "R6", "R7")
        assert src is not None
        assert src.render() == "osdAddr"

    def test_separated_by_unrelated_node(self):
        """One unrelated node between load and multiply start → still found."""
        from pseudo8051.ir.expr import Name
        unrelated = Assign(0x1001, Reg("R0"), Const(5))
        load = self._load_node("R6", "R7", Name("osdAddr"))
        mul_nodes = _make_mul16_nodes()
        nodes = [load, unrelated] + mul_nodes
        src = _find_pair_load_source(nodes, 2, "R6", "R7")
        assert src is not None
        assert src.render() == "osdAddr"

    def test_blocked_by_write(self):
        """Node that writes R6 between load and i → returns None."""
        from pseudo8051.ir.expr import Name
        load = self._load_node("R6", "R7", Name("osdAddr"))
        clobber = Assign(0x1001, Reg("R6"), Const(0))
        mul_nodes = _make_mul16_nodes()
        nodes = [load, clobber] + mul_nodes
        src = _find_pair_load_source(nodes, 2, "R6", "R7")
        assert src is None

    def test_no_load_returns_none(self):
        """No preceding load → returns None."""
        mul_nodes = _make_mul16_nodes()
        src = _find_pair_load_source(mul_nodes, 0, "R6", "R7")
        assert src is None

    def test_pair1_uses_load_source(self):
        """Full pattern with preceding R6R7=osdAddr load → pair1 renders as 'osdAddr'."""
        load = Assign(0x2000, RegGroup(("R6", "R7")), Name("osdAddr"))
        mul_nodes = _make_mul16_nodes(lo2=Const(9), hi2=Const(0))
        nodes = [load] + mul_nodes
        # match starting at index 1 (first mul node)
        result = Mul16Pattern().match(nodes, 1, {}, lambda ns, rm: ns)
        assert result is not None
        node = result[0][0]
        assert isinstance(node.rhs, BinOp)
        assert node.rhs.lhs.render() == "osdAddr"


class TestPair1XRAMAnnotation:
    """Annotation-based XRAM parent lookup for pair1 (loop-body scope case)."""

    def _make_ann_with_xram(self, rhi, rhi_sym, rlo, rlo_sym):
        ann = NodeAnnotation()
        ann.reg_exprs = {
            rhi: XRAMRef(Name(rhi_sym)),
            rlo: XRAMRef(Name(rlo_sym)),
        }
        return ann

    def test_xram_parent_used_as_pair1(self):
        """When nodes[i] has XRAM annotations for Rhi1/Rlo1 and reg_map has the
        parent VarInfo keyed by hi_sym, pair1 should render as the variable name."""
        vi_parent = VarInfo("osdAddr", "int16_t", (), xram_sym="EXT_DC41",
                            xram_addr=0xdc41)
        vi_lo_byte = VarInfo("osdAddr.lo", "uint8_t", (), xram_sym="EXT_DC42",
                             is_byte_field=True, xram_addr=0xdc42)
        reg_map = {
            "EXT_DC41": vi_parent,
            "_byte_EXT_DC42": vi_lo_byte,
        }
        mul_nodes = _make_mul16_nodes(lo2=Const(9), hi2=Const(0))
        ann = self._make_ann_with_xram("R6", "EXT_DC41", "R7", "EXT_DC42")
        mul_nodes[0].ann = ann  # annotate first node (A = R7)
        result = Mul16Pattern().match(mul_nodes, 0, reg_map, lambda ns, rm: ns)
        assert result is not None
        node = result[0][0]
        assert isinstance(node.rhs, BinOp)
        assert node.rhs.lhs.render() == "osdAddr"

    def test_byte_field_parent_not_used(self):
        """If reg_map[hi_sym] is a byte-field entry, it must not be used as pair1."""
        vi_byte = VarInfo("osdAddr.hi", "uint8_t", (), xram_sym="EXT_DC41",
                          is_byte_field=True, xram_addr=0xdc41)
        reg_map = {"EXT_DC41": vi_byte}
        mul_nodes = _make_mul16_nodes(lo2=Const(9), hi2=Const(0))
        ann = self._make_ann_with_xram("R6", "EXT_DC41", "R7", "EXT_DC42")
        mul_nodes[0].ann = ann
        result = Mul16Pattern().match(mul_nodes, 0, reg_map, lambda ns, rm: ns)
        assert result is not None
        node = result[0][0]
        # pair1 should fall back to R6R7 group substitution, not "osdAddr.hi"
        assert node.rhs.lhs.render() != "osdAddr.hi"

    def test_no_annotation_falls_back_to_reg_group(self):
        """No annotation → pair1 falls back to RegGroup substitution."""
        mul_nodes = _make_mul16_nodes(lo2=Const(9), hi2=Const(0))
        result = Mul16Pattern().match(mul_nodes, 0, {}, lambda ns, rm: ns)
        assert result is not None
        node = result[0][0]
        assert isinstance(node.rhs, BinOp)
        # Without reg_map alias, renders as the register pair name
        assert node.rhs.lhs.render() == "R6R7"
