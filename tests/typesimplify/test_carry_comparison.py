from pseudo8051.passes.typesimplify._post import _simplify_carry_comparison
from pseudo8051.ir.hir import Assign, CompoundAssign, WhileNode, ExprStmt, IfNode, NodeAnnotation
from pseudo8051.ir.expr import Reg, Regs, RegGroup, Name, Const, BinOp, Call
from pseudo8051.passes.patterns._utils import TypeGroup


def _ann_with_a_var(var_name: str, c_val: int = 0) -> NodeAnnotation:
    """Build a NodeAnnotation where A maps to var_name and C is known to be c_val."""
    ann = NodeAnnotation()
    ann.reg_groups = [TypeGroup(var_name, "uint8_t", ("A",), xram_sym="EXT_SOME")]
    ann.reg_consts = {"C": c_val}
    return ann


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


class TestCarryComparison8bit:
    """8-bit CLR C + SUBB comparison pattern in while(C) conditions."""

    def _make_subb8(self, var_name: str, operand, c_val: int = 0) -> CompoundAssign:
        """Return a CompoundAssign(A, '-=', operand) annotated as an 8-bit SUBB."""
        node = CompoundAssign(0x10, Reg("A"), "-=", operand)
        node.ann = _ann_with_a_var(var_name, c_val=c_val)
        return node

    # ── Positive cases ──────────────────────────────────────────────────────

    def test_const_operand_condition_replaced(self):
        """while(C) + A -= 6 (var1, C=0) → while (var1 < 6); SUBB removed."""
        subb = self._make_subb8("var1", Const(6))
        body = [subb, ExprStmt(0x20, Call("do_something", []))]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert isinstance(wn.condition, BinOp)
        assert wn.condition.op == "<"
        assert wn.condition.lhs == Name("var1")
        assert wn.condition.rhs == Const(6)
        # SUBB removed from body (C not used after it)
        assert len(wn.body_nodes) == 1
        assert wn.body_nodes[0] is body[1]

    def test_name_operand_condition_replaced(self):
        """while(C) + A -= var2 (var1, C=0) → while (var1 < var2); SUBB removed."""
        subb = self._make_subb8("var1", Name("var2"))
        body = [subb]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert wn.condition == BinOp(Name("var1"), "<", Name("var2"))
        assert len(wn.body_nodes) == 0

    def test_subb_with_plus_zero_tail(self):
        """A -= BinOp(6, '+', Const(0)) (propagation left +0 unfolded) → var1 < 6."""
        # _propagate_values substitutes C=0 → BinOp(6, '+', Const(0)) before
        # _simplify_arithmetic folds it.
        operand_with_zero = BinOp(Const(6), "+", Const(0))
        subb = self._make_subb8("var1", operand_with_zero)
        body = [subb]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert wn.condition.op == "<"
        assert wn.condition.lhs == Name("var1")
        # operand is lhs of the BinOp (the "+ 0" tail is stripped)
        assert wn.condition.rhs == Const(6)

    def test_subb_kept_when_c_used_by_inner_if(self):
        """When if(C) follows the SUBB, keep it in the body (its C side-effect is needed)."""
        subb = self._make_subb8("var1", Name("var2"))
        inner_if = IfNode(0x20, Reg("C"), [ExprStmt(0x30, Call("do_something", []))])
        body = [subb, inner_if]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert wn.condition == BinOp(Name("var1"), "<", Name("var2"))
        # SUBB is KEPT because if(C) needs C
        assert len(wn.body_nodes) == 2
        assert wn.body_nodes[0] is subb

    def test_subb_kept_when_c_written_then_used(self):
        """If C is re-written before being used, SUBB is safe to remove."""
        subb = self._make_subb8("var1", Const(6))
        # C is overwritten before use → safe to prune
        c_write = Assign(0x20, Reg("C"), Const(0))
        inner_if = IfNode(0x30, Reg("C"), [])
        body = [subb, c_write, inner_if]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert wn.condition.op == "<"
        # SUBB removed: C-write at 0x20 comes before any C-read
        assert len(wn.body_nodes) == 2
        assert wn.body_nodes[0] is c_write

    # ── Negative cases ──────────────────────────────────────────────────────

    def test_no_annotation_unchanged(self):
        """CompoundAssign with no annotation is not treated as comparison."""
        subb = CompoundAssign(0x10, Reg("A"), "-=", Const(6))
        # no ann set
        body = [subb]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert wn.condition == Reg("C")
        assert len(wn.body_nodes) == 1

    def test_c_not_zero_unchanged(self):
        """SUBB with C=1 (SETB C predecessor) is not the loop condition setter."""
        subb = self._make_subb8("var1", Const(6), c_val=1)  # C was 1, not CLR C
        body = [subb]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert wn.condition == Reg("C")
        assert len(wn.body_nodes) == 1

    def test_no_a_var_annotation_unchanged(self):
        """SUBB where A is not mapped to a named variable is left unchanged."""
        subb = CompoundAssign(0x10, Reg("A"), "-=", Const(6))
        ann = NodeAnnotation()
        ann.reg_consts = {"C": 0}
        ann.reg_groups = []   # no TypeGroup for A
        subb.ann = ann
        body = [subb]
        nodes = [WhileNode(0, Reg("C"), body)]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert wn.condition == Reg("C")

    def test_non_c_while_condition_unchanged(self):
        """A while(R7 != 0) loop is not touched."""
        subb = self._make_subb8("var1", Const(6))
        nodes = [WhileNode(0, BinOp(Reg("R7"), "!=", Const(0)), [subb])]
        result = _simplify_carry_comparison(nodes)
        wn = result[0]
        assert wn.condition == BinOp(Reg("R7"), "!=", Const(0))
