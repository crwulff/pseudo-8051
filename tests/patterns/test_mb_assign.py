from pseudo8051.ir.hir import Statement, Assign, ExprStmt, IfNode
from pseudo8051.ir.expr import Reg, UnaryOp, Name, Const
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

    def test_no_collapse_starts_at_lo(self):
        """Sequence starting with .lo is not a start of a new sequence → unchanged."""
        nodes = [
            Assign(0, Name("var0.lo"), Name("count.lo")),
            Assign(2, Name("var0.hi"), Name("count.hi")),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 2

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
            Statement(0, "foo();"),
            Assign(2, Name("var0.hi"), Name("count.hi")),
            _dptr_inc(3),
            Assign(4, Name("var0.lo"), Name("count.lo")),
            Statement(6, "bar();"),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 3
        assert result[0].text == "foo();"
        assert result[1].render()[0][1] == "var0 = count;"
        assert result[2].text == "bar();"
