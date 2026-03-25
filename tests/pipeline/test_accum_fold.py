from pseudo8051.ir.hir import Statement, IfGoto, Assign, CompoundAssign, IfNode
from pseudo8051.ir.expr import Reg, Const, BinOp, XRAMRef, Name


class TestAccumFoldPattern:
    """Test AccumFoldPattern directly (bypasses TypeAwareSimplifier reg_map guard)."""

    def _match(self, nodes):
        from pseudo8051.passes.patterns.accum_fold import AccumFoldPattern
        return AccumFoldPattern().match(nodes, 0, {}, lambda ns, rm: ns)

    def test_xram_load_mask_ifgoto(self):
        """
        DPTR=sym; A=XRAM[sym]; A&=1; IfGoto(A==0, label)
        → IfGoto((XRAM[sym] & 1) == 0, label)
        """
        nodes = [
            Assign(0x1000, Reg("DPTR"), Name("EXT_28")),
            Assign(0x1002, Reg("A"),    XRAMRef(Name("EXT_28"))),
            CompoundAssign(0x1004, Reg("A"), "&=", Const(1)),
            IfGoto(0x1006, BinOp(Reg("A"), "==", Const(0)), "label_1010"),
        ]
        result = self._match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        assert len(replacement) == 1
        node = replacement[0]
        assert isinstance(node, IfGoto)
        assert node.label == "label_1010"
        assert node.cond.render() == "(XRAM[EXT_28] & 1) == 0"

    def test_load_mask_ifnode(self):
        """
        A=XRAM[sym]; A&=1; IfNode(A!=0, [then_stmt])
        → IfNode((XRAM[sym] & 1) != 0, [then_stmt])
        """
        then_stmt = Statement(0x1010, "R7 = 1;")
        nodes = [
            Assign(0x1000, Reg("A"), XRAMRef(Name("SYM"))),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(1)),
            IfNode(0x1004, BinOp(Reg("A"), "!=", Const(0)), [then_stmt]),
        ]
        result = self._match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 3
        node = replacement[0]
        assert isinstance(node, IfNode)
        assert isinstance(node.condition, BinOp)
        assert node.condition.render() == "(XRAM[SYM] & 1) != 0"
        assert len(node.then_nodes) == 1

    def test_reg_load_mask_relay(self):
        """
        A=R7; A&=0xf0; Assign(R6, A)
        → Assign(R6, R7 & 0xf0)
        """
        nodes = [
            Assign(0x1000, Reg("A"),  Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(0xf0)),
            Assign(0x1004, Reg("R6"), Reg("A")),
        ]
        result = self._match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 3
        node = replacement[0]
        assert isinstance(node, Assign)
        assert node.lhs == Reg("R6")
        assert node.rhs.render() == "R7 & 0xf0"

    def test_ifgoto_no_compound(self):
        """
        A=XRAM[sym]; IfGoto(A!=0, label)
        → IfGoto(XRAM[sym] != 0, label)
        """
        nodes = [
            Assign(0x1000, Reg("A"), XRAMRef(Name("SYM"))),
            IfGoto(0x1002, BinOp(Reg("A"), "!=", Const(0)), "label_1020"),
        ]
        result = self._match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 2
        node = replacement[0]
        assert isinstance(node, IfGoto)
        assert node.label == "label_1020"
        assert node.cond.render() == "XRAM[SYM] != 0"

    def test_pure_relay_unaffected(self):
        """
        A=XRAM[sym]; Assign(R7, A) — no compound assigns, no DPTR prefix.
        AccumFoldPattern must return None.
        """
        nodes = [
            Assign(0x1000, Reg("A"),  XRAMRef(Name("SYM"))),
            Assign(0x1002, Reg("R7"), Reg("A")),
        ]
        result = self._match(nodes)
        assert result is None
