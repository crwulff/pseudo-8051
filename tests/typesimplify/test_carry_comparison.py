from pseudo8051.passes.typesimplify._post import _simplify_carry_comparison
from pseudo8051.ir.hir import Assign, CompoundAssign, WhileNode, ExprStmt
from pseudo8051.ir.expr import Reg, Regs, RegGroup, Name, Const, BinOp, Call


class TestCarryComparison:

    def _make_subb16(self, k=0):
        """Return the 4-node CLR C + SUBB lo + MOV A,Rhi + SUBB hi sequence."""
        return [
            Assign(k,   Reg("C"), Const(0)),
            CompoundAssign(k+1, Reg("A"), "-=", BinOp(Reg("R7"), "+", Reg("C"))),
            Assign(k+2, Reg("A"), Reg("R4")),
            CompoundAssign(k+3, Reg("A"), "-=", BinOp(Reg("R6"), "+", Reg("C"))),
        ]

    def test_carry_condition_replaced(self):
        """WhileNode(C) + SUBB16 body → while (offset < _count)."""
        body = [
            Assign(0, RegGroup(("R6", "R7")), Name("_count")),
            Assign(1, RegGroup(("R4", "R5")), Name("offset")),
        ] + self._make_subb16(k=2) + [
            ExprStmt(6, Call("do_something", [])),
        ]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert isinstance(wn, WhileNode)
        assert isinstance(wn.condition, BinOp)
        assert wn.condition.op == "<"
        assert wn.condition.lhs.name == "offset"
        assert wn.condition.rhs.name == "_count"
        # 4 SUBB nodes removed, 2 loads + 1 stmt remain
        assert len(wn.body_nodes) == 3

    def test_non_carry_condition_unchanged(self):
        """WhileNode with non-C condition is untouched."""
        nodes = [WhileNode(0, BinOp(Reg("R7"), "!=", Const(0)), [])]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert isinstance(wn.condition, BinOp) and wn.condition.op == "!="

    def test_missing_reggroup_leaves_unchanged(self):
        """If no RegGroup assigns found, body and condition are left unchanged."""
        body = self._make_subb16(k=0) + [ExprStmt(4, Call("do_something", []))]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        # condition still C (couldn't resolve names)
        assert isinstance(wn.condition, Regs) and wn.condition.is_single and wn.condition.name == "C"
        assert len(wn.body_nodes) == 5  # unchanged
