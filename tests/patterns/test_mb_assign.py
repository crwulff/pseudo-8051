from pseudo8051.ir.hir import Assign, CompoundAssign, ExprStmt, IfNode
from pseudo8051.ir.expr import Reg, Regs, UnaryOp, Name, Const, BinOp, Call, RegGroup
from pseudo8051.passes.patterns.mb_assign import collapse_mb_assigns


def _dptr_inc(ea=0):
    return ExprStmt(ea, UnaryOp("++", Reg("DPTR"), post=True))


class TestCollapseMbAssigns:
    def test_two_byte_field_var_src(self):
        """var0.hi = count.hi; DPTR++; var0.lo = count.lo; → var0 = count;"""
        nodes = [
            Assign(0, Name("var0.hi"), Name("count.hi")),
            _dptr_inc(),
            Assign(2, Name("var0.lo"), Name("count.lo")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "var0 = count;"

    def test_two_byte_field_const_src_name(self):
        """var0.hi = Name("0x12"); DPTR++; var0.lo = Name("0x34"); → var0 = 0x1234;
        (XRAMLocalWritePattern wraps constants in Name)"""
        nodes = [
            Assign(0, Name("var0.hi"), Name("0x12")),
            _dptr_inc(),
            Assign(2, Name("var0.lo"), Name("0x34")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "var0 = 0x1234;"

    def test_two_byte_field_const_src_const(self):
        """var0.hi = Const(0x12); DPTR++; var0.lo = Const(0x34); → var0 = 0x1234;
        (AccumRelayPattern may produce Const nodes directly)"""
        nodes = [
            Assign(0, Name("var0.hi"), Const(0x12)),
            _dptr_inc(),
            Assign(2, Name("var0.lo"), Const(0x34)),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "var0 = 0x1234;"

    def test_no_collapse_mismatched_rhs(self):
        """Different RHS parents → no collapse."""
        nodes = [
            Assign(0, Name("var0.hi"), Name("count.hi")),
            Assign(2, Name("var0.lo"), Name("other.lo")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 2

    def test_no_collapse_single_byte(self):
        """Single byte-field assignment (no partner) → unchanged."""
        nodes = [Assign(0, Name("var0.hi"), Name("count.hi"))]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert result[0].render()[0][1] == "var0.hi = count.hi;"

    def test_collapse_starts_at_lo(self):
        """Sequence starting with .lo (lo-first order) also collapses."""
        nodes = [
            Assign(0, Name("var0.lo"), Name("count.lo")),
            Assign(2, Name("var0.hi"), Name("count.hi")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].lhs == Name("var0")
        assert result[0].rhs == Name("count")

    def test_four_byte_field_var_src(self):
        """b0..b3 sequence collapses to single assignment."""
        nodes = [
            Assign(0, Name("result.b0"), Name("src.b0")),
            Assign(2, Name("result.b1"), Name("src.b1")),
            Assign(4, Name("result.b2"), Name("src.b2")),
            Assign(6, Name("result.b3"), Name("src.b3")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "result = src;"

    def test_recurse_into_if_node(self):
        """Byte-field assignments inside IfNode bodies are also collapsed."""
        inner = [
            Assign(0, Name("var0.hi"), Name("count.hi")),
            Assign(2, Name("var0.lo"), Name("count.lo")),
        ]
        if_node = IfNode(0, "R3 == 0", inner, [])
        nodes = [if_node]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], IfNode)
        assert len(result[0].then_nodes) == 1
        assert result[0].then_nodes[0].render()[0][1] == "var0 = count;"

    def test_surrounding_nodes_preserved(self):
        """Other statements before/after the sequence are preserved."""
        nodes = [
            ExprStmt(0, Call("foo", [])),
            Assign(2, Name("var0.hi"), Name("count.hi")),
            _dptr_inc(3),
            Assign(4, Name("var0.lo"), Name("count.lo")),
            ExprStmt(6, Call("bar", [])),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 3
        assert result[0].render()[0][1] == "foo();"
        assert result[1].render()[0][1] == "var0 = count;"
        assert result[2].render()[0][1] == "bar();"


def _reg_pair(hi, lo):
    """Multi-register Regs (brace=False, like R6R7)."""
    return Regs((hi, lo))


def _c():
    """Reg('C') — carry flag."""
    return Reg('C')


class TestPairStore:
    """Multi-reg result → byte-field store collapse:
       R6R7 = expr; var.hi = R6; var.lo = R7  →  var = expr
    """

    def test_basic_hi_lo_order(self):
        """R6R7 = expr; var.hi = R6; var.lo = R7 → var = expr"""
        expr = BinOp(Name("osdAddr"), "*", Const(9))
        nodes = [
            Assign(0, _reg_pair("R6", "R7"), expr),
            Assign(2, Name("osdAddr.hi"), Reg("R6")),
            Assign(4, Name("osdAddr.lo"), Reg("R7")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "osdAddr = osdAddr * 9;"

    def test_basic_lo_hi_order(self):
        """R6R7 = expr; var.lo = R7; var.hi = R6 → var = expr (lo-first store)"""
        expr = BinOp(Name("osdAddr"), "*", Const(9))
        nodes = [
            Assign(0, _reg_pair("R6", "R7"), expr),
            Assign(2, Name("osdAddr.lo"), Reg("R7")),
            Assign(4, Name("osdAddr.hi"), Reg("R6")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "osdAddr = osdAddr * 9;"

    def test_source_nodes_set(self):
        """Collapsed result has source_nodes covering all consumed nodes."""
        n0 = Assign(0, _reg_pair("R6", "R7"), BinOp(Name("v"), "*", Const(3)))
        n1 = Assign(2, Name("v.hi"), Reg("R6"))
        n2 = Assign(4, Name("v.lo"), Reg("R7"))
        result = collapse_mb_assigns([n0, n1, n2])
        assert len(result) == 1
        assert n0 in result[0].source_nodes
        assert n1 in result[0].source_nodes
        assert n2 in result[0].source_nodes

    def test_with_following_compound_assign(self):
        """R6R7 = v*9; v.hi = R6; v.lo = R7; v += x → v = v*9 + x"""
        expr = BinOp(Name("v"), "*", Const(9))
        nodes = [
            Assign(0, _reg_pair("R6", "R7"), expr),
            Assign(2, Name("v.hi"), Reg("R6")),
            Assign(4, Name("v.lo"), Reg("R7")),
            CompoundAssign(6, Name("v"), "+=", Name("x")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "v = v * 9 + x;"

    def test_with_dptr_inc_between(self):
        """DPTR++ between stores is silently consumed."""
        expr = BinOp(Name("v"), "*", Const(3))
        nodes = [
            Assign(0, _reg_pair("R4", "R5"), expr),
            ExprStmt(1, UnaryOp("++", Reg("DPTR"), post=True)),
            Assign(2, Name("v.hi"), Reg("R4")),
            Assign(4, Name("v.lo"), Reg("R5")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert result[0].render()[0][1] == "v = v * 3;"

    def test_wrong_register_not_matched(self):
        """Byte-field stores with wrong registers are not collapsed."""
        nodes = [
            Assign(0, _reg_pair("R6", "R7"), BinOp(Name("v"), "*", Const(9))),
            Assign(2, Name("v.hi"), Reg("R4")),   # wrong: R4, not R6
            Assign(4, Name("v.lo"), Reg("R7")),
        ]
        result = collapse_mb_assigns(nodes)
        # pair-store fails; nodes fall through to normal processing
        assert len(result) == 3

    def test_mismatched_var_names_not_collapsed(self):
        """hi store for one var and lo store for another → no collapse."""
        nodes = [
            Assign(0, _reg_pair("R6", "R7"), Name("x")),
            Assign(2, Name("a.hi"), Reg("R6")),
            Assign(4, Name("b.lo"), Reg("R7")),   # different parent
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 3


class TestAddCarryPair:
    """16-bit add-with-carry: var.lo = var.lo + x; var.hi = var.hi + y + C; → var += combined"""

    def _make_nodes(self, lo_rhs, hi_rhs):
        return [
            Assign(0, Name("v.lo"), lo_rhs),
            Assign(2, Name("v.hi"), hi_rhs),
        ]

    def test_name_addend_lo_first(self):
        """v.lo = v.lo + other.lo; v.hi = (v.hi + other.hi) + C; → v += other"""
        nodes = self._make_nodes(
            BinOp(Name("v.lo"), "+", Name("other.lo")),
            BinOp(BinOp(Name("v.hi"), "+", Name("other.hi")), "+", _c()),
        )
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], CompoundAssign)
        assert result[0].render()[0][1] == "v += other;"

    def test_name_addend_hi_first(self):
        """hi before lo order: v.hi = (v.hi + other.hi) + C; v.lo = v.lo + other.lo; → v += other"""
        nodes = [
            Assign(0, Name("v.hi"), BinOp(BinOp(Name("v.hi"), "+", Name("other.hi")), "+", _c())),
            Assign(2, Name("v.lo"), BinOp(Name("v.lo"), "+", Name("other.lo"))),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], CompoundAssign)
        assert result[0].render()[0][1] == "v += other;"

    def test_const_addend(self):
        """v.lo = v.lo + 5; v.hi = (v.hi + 0) + C; → v += 5"""
        nodes = self._make_nodes(
            BinOp(Name("v.lo"), "+", Const(5)),
            BinOp(BinOp(Name("v.hi"), "+", Const(0)), "+", _c()),
        )
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], CompoundAssign)
        assert result[0].render()[0][1] == "v += 5;"

    def test_const_addend_hi_nonzero(self):
        """v.lo = v.lo + 0x34; v.hi = (v.hi + 0x12) + C; → v += 0x1234"""
        nodes = self._make_nodes(
            BinOp(Name("v.lo"), "+", Const(0x34)),
            BinOp(BinOp(Name("v.hi"), "+", Const(0x12)), "+", _c()),
        )
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], CompoundAssign)
        assert result[0].render()[0][1] == "v += 0x1234;"

    def test_case_b_carry_in_tail(self):
        """Case B: v.hi = v.hi + (other.hi + C); v.lo = v.lo + other.lo; → v += other"""
        nodes = self._make_nodes(
            BinOp(Name("v.lo"), "+", Name("other.lo")),
            BinOp(Name("v.hi"), "+", BinOp(Name("other.hi"), "+", _c())),
        )
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], CompoundAssign)
        assert result[0].render()[0][1] == "v += other;"

    def test_no_carry_not_matched(self):
        """v.hi = v.hi + other.hi (no carry) → not collapsed as add-carry"""
        nodes = self._make_nodes(
            BinOp(Name("v.lo"), "+", Name("other.lo")),
            BinOp(Name("v.hi"), "+", Name("other.hi")),
        )
        # No carry: should not collapse to add-carry (field-copy check also fails)
        result = collapse_mb_assigns(nodes)
        assert len(result) == 2

    def test_carry_in_lo_not_matched(self):
        """Carry in lo_rhs → not a valid add-carry (lo should be carry-free)"""
        nodes = self._make_nodes(
            BinOp(Name("v.lo"), "+", BinOp(Name("other.lo"), "+", _c())),
            BinOp(BinOp(Name("v.hi"), "+", Name("other.hi")), "+", _c()),
        )
        result = collapse_mb_assigns(nodes)
        assert len(result) == 2

    def test_mismatched_addend_parents_not_combined(self):
        """lo addend from 'a.lo' and hi addend from 'b.hi' → cannot combine"""
        nodes = self._make_nodes(
            BinOp(Name("v.lo"), "+", Name("a.lo")),
            BinOp(BinOp(Name("v.hi"), "+", Name("b.hi")), "+", _c()),
        )
        result = collapse_mb_assigns(nodes)
        assert len(result) == 2
