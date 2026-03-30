from pseudo8051.ir.hir import Statement, Assign, ExprStmt, IfNode
from pseudo8051.ir.expr import Reg, UnaryOp
from pseudo8051.passes.patterns.mb_assign import collapse_mb_assigns


class TestCollapseMbAssigns:
    def test_two_byte_field_var_src(self):
        """var0.hi = count.hi; DPTR++; var0.lo = count.lo; → var0 = count;"""
        dptr_inc = ExprStmt(0, UnaryOp("++", Reg("DPTR"), post=True))
        nodes = [
            Statement(0, "var0.hi = count.hi;"),
            dptr_inc,
            Statement(2, "var0.lo = count.lo;"),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "var0 = count;"

    def test_two_byte_field_const_src(self):
        """var0.hi = 0x12; DPTR++; var0.lo = 0x34; → var0 = 0x1234;"""
        dptr_inc = Statement(0, "DPTR++;")
        nodes = [
            Statement(0, "var0.hi = 0x12;"),
            dptr_inc,
            Statement(2, "var0.lo = 0x34;"),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "var0 = 0x1234;"

    def test_no_collapse_mismatched_rhs(self):
        """Different RHS parents → no collapse."""
        nodes = [
            Statement(0, "var0.hi = count.hi;"),
            Statement(2, "var0.lo = other.lo;"),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 2

    def test_no_collapse_single_byte(self):
        """Single byte-field statement (no partner) → unchanged."""
        nodes = [Statement(0, "var0.hi = count.hi;")]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert result[0].text == "var0.hi = count.hi;"

    def test_no_collapse_starts_at_lo(self):
        """Sequence starting with .lo is not a start of a new sequence → unchanged."""
        nodes = [
            Statement(0, "var0.lo = count.lo;"),
            Statement(2, "var0.hi = count.hi;"),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 2

    def test_four_byte_field_var_src(self):
        """b0..b3 sequence collapses to single assignment."""
        nodes = [
            Statement(0, "result.b0 = src.b0;"),
            Statement(2, "result.b1 = src.b1;"),
            Statement(4, "result.b2 = src.b2;"),
            Statement(6, "result.b3 = src.b3;"),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)
        assert result[0].render()[0][1] == "result = src;"

    def test_recurse_into_if_node(self):
        """Byte-field assignments inside IfNode bodies are also collapsed."""
        inner = [
            Statement(0, "var0.hi = count.hi;"),
            Statement(2, "var0.lo = count.lo;"),
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
        dptr_inc = ExprStmt(0, UnaryOp("++", Reg("DPTR"), post=True))
        nodes = [
            Statement(0, "foo();"),
            Statement(2, "var0.hi = count.hi;"),
            dptr_inc,
            Statement(4, "var0.lo = count.lo;"),
            Statement(6, "bar();"),
        ]
        result = collapse_mb_assigns(nodes)
        assert len(result) == 3
        assert result[0].text == "foo();"
        assert result[1].render()[0][1] == "var0 = count;"
        assert result[2].text == "bar();"
