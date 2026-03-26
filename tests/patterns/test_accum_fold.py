"""
tests/patterns/test_accum_fold.py — AccumFoldPattern edge cases:
  - A += A → A * 2 normalization (issue 2.2)
  - MUL AB folding (issue 2.1)
  - combined chains
"""

from pseudo8051.ir.hir import Assign, CompoundAssign, IfGoto, ReturnStmt
from pseudo8051.ir.expr import Reg, Const, BinOp, XRAMRef, Name, RegGroup, Cast


def _match(nodes):
    from pseudo8051.passes.patterns.accum_fold import AccumFoldPattern
    return AccumFoldPattern().match(nodes, 0, {}, lambda ns, rm: ns)


class TestAccumFoldADoubling:
    def test_a_plus_a_normalize_ifgoto(self):
        """A = R7; A += A; IfGoto(A == 0, lbl) → IfGoto(R7 * 2 == 0, lbl)"""
        nodes = [
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Reg("A")),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_x"),
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 3
        assert len(replacement) == 1
        node = replacement[0]
        assert isinstance(node, IfGoto)
        rendered = node.cond.render()
        assert "R7" in rendered
        assert "2" in rendered
        assert "==" in rendered

    def test_a_plus_a_in_chain(self):
        """A = R7; A += A; A &= 0xfe; IfGoto(A == 0, lbl) → condition has R7*2 & 0xfe"""
        nodes = [
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Reg("A")),
            CompoundAssign(0x1004, Reg("A"), "&=", Const(0xfe)),
            IfGoto(0x1006, BinOp(Reg("A"), "==", Const(0)), "label_x"),
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        node = replacement[0]
        assert isinstance(node, IfGoto)
        rendered = node.cond.render()
        assert "R7" in rendered
        assert "2" in rendered
        assert "0xfe" in rendered
        assert "==" in rendered

    def test_a_plus_a_return(self):
        """A = R7; A += A; ReturnStmt(A) → ReturnStmt(R7 * 2)"""
        nodes = [
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Reg("A")),
            ReturnStmt(0x1004, Reg("A")),
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 3
        node = replacement[0]
        assert isinstance(node, ReturnStmt)
        assert node.value.render() == "R7 * 2"


class TestAccumFoldMulAB:
    def test_mul_ab_ifgoto(self):
        """A = XRAM[sym]; B = 4; {B,A} = A*B; IfGoto(A == 0, lbl) → IfGoto((uint8_t)(XRAM[sym] * 4) == 0, lbl)"""
        nodes = [
            Assign(0x1000, Reg("A"), XRAMRef(Name("SYM"))),
            Assign(0x1002, Reg("B"), Const(4)),
            Assign(0x1004, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            IfGoto(0x1006, BinOp(Reg("A"), "==", Const(0)), "label_y"),
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        node = replacement[0]
        assert isinstance(node, IfGoto)
        rendered = node.cond.render()
        assert "uint8_t" in rendered
        assert "4" in rendered
        assert "SYM" in rendered

    def test_mul_ab_return(self):
        """A = R7; B = R5; {B,A} = A*B; ReturnStmt(A) → ReturnStmt((uint8_t)(R7 * R5))"""
        nodes = [
            Assign(0x1000, Reg("A"), Reg("R7")),
            Assign(0x1002, Reg("B"), Reg("R5")),
            Assign(0x1004, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            ReturnStmt(0x1006, Reg("A")),
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        node = replacement[0]
        assert isinstance(node, ReturnStmt)
        rendered = node.value.render()
        assert "uint8_t" in rendered
        assert "R7" in rendered
        assert "R5" in rendered

    def test_mul_ab_with_prior_compound(self):
        """A = R7; A &= 0x0f; B = 4; {B,A} = A*B; ReturnStmt(A)"""
        nodes = [
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(0x0f)),
            Assign(0x1004, Reg("B"), Const(4)),
            Assign(0x1006, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            ReturnStmt(0x1008, Reg("A")),
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 5
        node = replacement[0]
        assert isinstance(node, ReturnStmt)
        rendered = node.value.render()
        assert "uint8_t" in rendered
        assert "0xf" in rendered   # Const(0x0f) renders as 0xf
        assert "4" in rendered
