from pseudo8051.passes.typesimplify._post import _simplify_orl_zero_check
from pseudo8051.ir.hir import CompoundAssign, IfNode, IfGoto, ExprStmt
from pseudo8051.ir.expr import Reg, Name, Const, BinOp, Call


class TestOrlZeroCheck:

    def test_orl_hi_before_ifnode_retval(self):
        """A |= count.hi + if (retval != 0) → if (count != 0)."""
        nodes = [
            CompoundAssign(0, Reg("A"), "|=", Name("count.hi")),
            IfNode(1, BinOp(Name("retval"), "!=", Const(0)),
                   [ExprStmt(2, Call("do_thing", []))], []),
        ]
        result = _simplify_orl_zero_check(nodes)
        assert len(result) == 1
        wn = result[0]
        assert isinstance(wn, IfNode)
        assert isinstance(wn.condition, BinOp)
        assert wn.condition.op == "!="
        assert wn.condition.lhs.name == "count"
        assert wn.condition.rhs.value == 0
        assert len(wn.then_nodes) == 1

    def test_orl_lo_before_ifnode_a(self):
        """A |= buf.lo + if (A == 0) → if (buf == 0).  Condition uses Reg("A")."""
        nodes = [
            CompoundAssign(0, Reg("A"), "|=", Name("buf.lo")),
            IfNode(1, BinOp(Reg("A"), "==", Const(0)), [], []),
        ]
        result = _simplify_orl_zero_check(nodes)
        assert len(result) == 1
        wn = result[0]
        assert isinstance(wn.condition, BinOp)
        assert wn.condition.op == "=="
        assert wn.condition.lhs.name == "buf"

    def test_no_byte_suffix_unchanged(self):
        """A |= count (no .hi/.lo suffix) → unchanged."""
        nodes = [
            CompoundAssign(0, Reg("A"), "|=", Name("count")),
            IfNode(1, BinOp(Name("retval"), "!=", Const(0)), [], []),
        ]
        result = _simplify_orl_zero_check(nodes)
        assert len(result) == 2  # both nodes preserved

    def test_non_zero_cond_unchanged(self):
        """A |= count.hi; if (x > 5) → unchanged (not a zero check)."""
        nodes = [
            CompoundAssign(0, Reg("A"), "|=", Name("count.hi")),
            IfNode(1, BinOp(Name("retval"), ">", Const(5)), [], []),
        ]
        result = _simplify_orl_zero_check(nodes)
        assert len(result) == 2  # unchanged

    def test_other_node_in_between_unchanged(self):
        """ORL not immediately before IfNode → unchanged."""
        nodes = [
            CompoundAssign(0, Reg("A"), "|=", Name("count.hi")),
            ExprStmt(1, Call("something_else", [])),
            IfNode(2, BinOp(Name("retval"), "!=", Const(0)), [], []),
        ]
        result = _simplify_orl_zero_check(nodes)
        assert len(result) == 3  # unchanged
