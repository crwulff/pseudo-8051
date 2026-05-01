from pseudo8051.passes.typesimplify._post import _simplify_carry_comparison, _simplify_subb_jc
from pseudo8051.ir.hir import Assign, CompoundAssign, WhileNode, ExprStmt, IfNode, IfGoto, NodeAnnotation
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


class TestSubbJcFold:
    """Tests for _simplify_subb_jc: CLR-C + SUBB + JC/JNC → typed comparison."""

    def _make_subb(self, var_name: str, operand, c_val: int = 0,
                   a_via_reg: str = None) -> CompoundAssign:
        """Build a SUBB CompoundAssign with annotation.

        If a_via_reg is given, the TypeGroup is on that register (e.g. 'R7'),
        and reg_exprs["A"] = Regs((a_via_reg,)) to model the A=Rx indirection.
        """
        node = CompoundAssign(0x10, Reg("A"), "-=", BinOp(operand, "+", Reg("C")))
        ann = NodeAnnotation()
        if a_via_reg:
            ann.reg_groups = [TypeGroup(var_name, "uint8_t", (a_via_reg,), xram_sym="EXT_SOME")]
            ann.reg_exprs = {"A": Regs((a_via_reg,))}
        else:
            ann.reg_groups = [TypeGroup(var_name, "uint8_t", ("A",), xram_sym="EXT_SOME")]
        ann.reg_consts = {"C": c_val}
        node.ann = ann
        return node

    def test_jc_becomes_less_than(self):
        """CLR C + SUBB A, #2 + if(C){} → if(var < 2){}"""
        subb = self._make_subb("flags", Const(2))
        if_node = IfNode(0x12, Reg("C"), [Assign(0x14, Reg("R0"), Const(1))])
        result = _simplify_subb_jc([subb, if_node])
        assert len(result) == 1
        assert isinstance(result[0], IfNode)
        assert result[0].condition == BinOp(Name("flags"), "<", Const(2))

    def test_jnc_becomes_greater_equal(self):
        """CLR C + SUBB A, #2 + if(!C){} → if(flags >= 2){}"""
        from pseudo8051.ir.expr import UnaryOp
        subb = self._make_subb("flags", Const(2))
        if_node = IfNode(0x12, UnaryOp("!", Reg("C")), [Assign(0x14, Reg("R0"), Const(1))])
        result = _simplify_subb_jc([subb, if_node])
        assert len(result) == 1
        assert isinstance(result[0], IfNode)
        assert result[0].condition == BinOp(Name("flags"), ">=", Const(2))

    def test_ifgoto_jc(self):
        """Works with IfGoto(C) as well."""
        subb = self._make_subb("flags", Const(3))
        goto = IfGoto(0x12, Reg("C"), "label_target")
        result = _simplify_subb_jc([subb, goto])
        assert len(result) == 1
        assert isinstance(result[0], IfGoto)
        assert result[0].cond == BinOp(Name("flags"), "<", Const(3))

    def test_no_a_var_annotation_unchanged(self):
        """SUBB where A is not mapped to a named variable is left unchanged."""
        node = CompoundAssign(0x10, Reg("A"), "-=", Const(2))
        ann = NodeAnnotation()
        ann.reg_consts = {"C": 0}
        ann.reg_groups = []   # no TypeGroup for A
        node.ann = ann
        if_node = IfNode(0x12, Reg("C"), [])
        result = _simplify_subb_jc([node, if_node])
        assert len(result) == 2
        assert result[1].condition == Reg("C")

    def test_c_unknown_unchanged(self):
        """SUBB with no C annotation is left unchanged."""
        node = CompoundAssign(0x10, Reg("A"), "-=", Const(2))
        ann = NodeAnnotation()
        ann.reg_groups = [TypeGroup("flags", "uint8_t", ("A",), xram_sym="EXT_SOME")]
        ann.reg_consts = {}   # C unknown
        node.ann = ann
        if_node = IfNode(0x12, Reg("C"), [])
        result = _simplify_subb_jc([node, if_node])
        assert len(result) == 2
        assert result[1].condition == Reg("C")

    def test_setb_c_subb_jc(self):
        """SETB C + SUBB A, #0 + if(C): A -= 0+1; C set when _timeout < 1."""
        from pseudo8051.ir.hir import ExprStmt
        from pseudo8051.ir.expr import UnaryOp
        # Simulates: A = _timeout; --_timeout; A -= 0+1; if (C) {}
        a_load = Assign(0x0e, Reg("A"), Name("_timeout"))
        dec_stmt = ExprStmt(0x0f, UnaryOp("--", Name("_timeout"), post=False))
        subb = CompoundAssign(0x10, Reg("A"), "-=", BinOp(Const(0), "+", Const(1)))
        ann = NodeAnnotation()
        ann.reg_groups = []
        ann.reg_consts = {"C": 1}
        subb.ann = ann
        if_node = IfNode(0x12, Reg("C"), [])
        result = _simplify_subb_jc([a_load, dec_stmt, subb, if_node])
        assert len(result) == 3   # a_load + dec_stmt kept, if_node replaced
        assert result[0] is a_load
        assert result[1] is dec_stmt
        assert isinstance(result[2], IfNode)
        assert result[2].condition == BinOp(Name("_timeout"), "<", Const(1))

    def test_a_via_reg_exprs(self):
        """A=R7 in reg_exprs + TypeGroup(R7=flags) → fold uses 'flags'."""
        subb = self._make_subb("flags", Const(2), a_via_reg="R7")
        if_node = IfNode(0x12, Reg("C"), [Assign(0x14, Reg("R0"), Const(1))])
        result = _simplify_subb_jc([subb, if_node])
        assert len(result) == 1
        assert isinstance(result[0], IfNode)
        assert result[0].condition == BinOp(Name("flags"), "<", Const(2))

    def test_a_via_preceding_xram_local_write(self):
        """Preceding Assign(Name(_var), Name(flags)) gives A's name via backward scan."""
        # Simulates: _var1 = flags; A -= 2; if (C) {}
        # where A held flags but no TypeGroup for A or R7 is present.
        xram_write = Assign(0x0e, Name("_var1"), Name("flags"))
        subb = CompoundAssign(0x10, Reg("A"), "-=", BinOp(Const(2), "+", Reg("C")))
        ann = NodeAnnotation()
        ann.reg_groups = []   # no TypeGroup for A or any register
        ann.reg_consts = {"C": 0}
        subb.ann = ann
        if_node = IfNode(0x12, Reg("C"), [Assign(0x14, Reg("R0"), Const(1))])
        result = _simplify_subb_jc([xram_write, subb, if_node])
        assert len(result) == 2   # xram_write kept, if_node replaced
        assert result[0] is xram_write
        assert isinstance(result[1], IfNode)
        assert result[1].condition == BinOp(Name("flags"), "<", Const(2))
