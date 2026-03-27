"""
tests/patterns/test_accum_fold.py — AccumFoldPattern edge cases:
  - A += A → A * 2 normalization (issue 2.2)
  - MUL AB folding (issue 2.1)
  - combined chains
"""

from pseudo8051.ir.hir import Assign, CompoundAssign, ExprStmt, IfGoto, ReturnStmt
from pseudo8051.ir.expr import Reg, Const, BinOp, UnaryOp, XRAMRef, Name, RegGroup, Cast


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


class TestMulABPairLookahead:
    def test_basic_mul_pair_no_lo_mod(self):
        """A=XRAM[X]; A&=0xF0; B=0x10; {B,A}=A*B; R7=A; A=B; R2=A; A=R7; R3=A → R2R3 = product"""
        nodes = [
            Assign(0x1000, Reg("A"), XRAMRef(Name("X"))),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(0xF0)),
            Assign(0x1004, Reg("B"), Const(0x10)),
            Assign(0x1006, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            Assign(0x1008, Reg("R7"), Reg("A")),
            Assign(0x100a, Reg("A"), Reg("B")),
            Assign(0x100c, Reg("R2"), Reg("A")),
            Assign(0x100e, Reg("A"), Reg("R7")),
            Assign(0x1010, Reg("R3"), Reg("A")),
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 9
        assert len(replacement) == 1
        node = replacement[0]
        assert isinstance(node, Assign)
        assert isinstance(node.lhs, RegGroup)
        assert set(node.lhs.regs) == {"R2", "R3"}
        rendered = node.rhs.render()
        assert "X" in rendered
        assert "0xf0" in rendered
        assert "0x10" in rendered
        # No Cast — full product (no uint8_t truncation)
        assert "uint8_t" not in rendered

    def test_mul_pair_with_or_modification(self):
        """A=XRAM[X]; A&=0xF0; B=0x10; {B,A}=A*B; R7=A; A=B; R2=A; A=R7; A|=XRAM[Y]; R3=A → R2R3 = product | Y"""
        nodes = [
            Assign(0x1000, Reg("A"), XRAMRef(Name("X"))),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(0xF0)),
            Assign(0x1004, Reg("B"), Const(0x10)),
            Assign(0x1006, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            Assign(0x1008, Reg("R7"), Reg("A")),
            Assign(0x100a, Reg("A"), Reg("B")),
            Assign(0x100c, Reg("R2"), Reg("A")),
            Assign(0x100e, Reg("A"), Reg("R7")),
            CompoundAssign(0x1010, Reg("A"), "|=", XRAMRef(Name("Y"))),
            Assign(0x1012, Reg("R3"), Reg("A")),
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 10
        assert len(replacement) == 1
        node = replacement[0]
        assert isinstance(node, Assign)
        assert isinstance(node.lhs, RegGroup)
        assert set(node.lhs.regs) == {"R2", "R3"}
        rendered = node.rhs.render()
        assert "X" in rendered
        assert "0xf0" in rendered
        assert "0x10" in rendered
        assert "Y" in rendered
        assert "|" in rendered
        assert "uint8_t" not in rendered

    def test_non_adjacent_pair_does_not_fold(self):
        """A=XRAM[X]; B=0x10; {B,A}=A*B; R7=A; A=B; R3=A; A=R7; R5=A → no fold (R3/R5 not adjacent)"""
        nodes = [
            Assign(0x1000, Reg("A"), XRAMRef(Name("X"))),
            Assign(0x1002, Reg("B"), Const(0x10)),
            Assign(0x1004, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            Assign(0x1006, Reg("R7"), Reg("A")),
            Assign(0x1008, Reg("A"), Reg("B")),
            Assign(0x100a, Reg("R3"), Reg("A")),
            Assign(0x100c, Reg("A"), Reg("R7")),
            Assign(0x100e, Reg("R5"), Reg("A")),
        ]
        result = _match(nodes)
        # Lookahead should return None; falls back to normal terminal handling
        # R7=A is the terminal (num_compound=1), so result is R7 = (uint8_t)(XRAM[X]*0x10)
        assert result is not None
        replacement, new_i = result
        assert new_i == 4  # consumed up to and including R7=A
        node = replacement[0]
        assert isinstance(node, Assign)
        assert node.lhs == Reg("R7")


class TestMulABPairLookaheadInterleaved:
    def test_interleaved_exprstmt_and_assigns(self):
        """Interleaved ExprStmt(DPTR++) + Assign(A, XRAM) + Assign(R5, A) between R7=A and A=B."""
        nodes = [
            Assign(0x1000, Reg("A"), Reg("R1")),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(0xF0)),
            Assign(0x1004, Reg("B"), Const(0x10)),
            Assign(0x1006, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            Assign(0x1008, Reg("R7"), Reg("A")),                              # terminal: lo save
            ExprStmt(0x100a, UnaryOp("++", Reg("DPTR"), post=True)),          # DPTR++ (safe)
            Assign(0x100c, Reg("A"), XRAMRef(Name("dc6b_sym"))),              # A = XRAM[..] (safe: lhs=A, but A=B not yet)
            Assign(0x100e, Reg("R5"), Reg("A")),                              # R5 = A (safe)
            Assign(0x1010, Reg("A"), Reg("B")),                               # A = B  ← found here
            Assign(0x1012, Reg("R2"), Reg("A")),                              # R2 = A (hi)
            Assign(0x1014, Reg("A"), Reg("R7")),                              # A = R7 (restore lo)
            CompoundAssign(0x1016, Reg("A"), "|=", Reg("R5")),               # A |= R5
            Assign(0x1018, Reg("R3"), Reg("A")),                              # R3 = A (lo final)
        ]
        result = _match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 13
        assert len(replacement) == 1
        node = replacement[0]
        assert isinstance(node, Assign)
        assert isinstance(node.lhs, RegGroup)
        assert set(node.lhs.regs) == {"R2", "R3"}
        rendered = node.rhs.render()
        assert "R1" in rendered
        assert "0xf0" in rendered
        assert "0x10" in rendered
        # R5 was reassigned to XRAM[dc6b_sym] in the interleaved block;
        # the rendered output must use that value, not the bare register name.
        assert "dc6b_sym" in rendered
        assert "R5" not in rendered
        assert "|" in rendered
        assert "uint8_t" not in rendered

    def test_b_clobbered_in_intermediate_no_fold(self):
        """B overwritten before A=B → lookahead aborts → falls back to lo-byte terminal."""
        nodes = [
            Assign(0x1000, Reg("A"), Reg("R1")),
            Assign(0x1002, Reg("B"), Const(0x10)),
            Assign(0x1004, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            Assign(0x1006, Reg("R7"), Reg("A")),                              # terminal: lo save
            Assign(0x1008, Reg("B"), Const(5)),                               # B clobbered → abort
            Assign(0x100a, Reg("A"), Reg("B")),
            Assign(0x100c, Reg("R2"), Reg("A")),
            Assign(0x100e, Reg("A"), Reg("R7")),
            Assign(0x1010, Reg("R3"), Reg("A")),
        ]
        result = _match(nodes)
        # Lookahead aborts; terminal is R7=A → R7 = (uint8_t)(R1 * 0x10)
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        node = replacement[0]
        assert isinstance(node, Assign)
        assert node.lhs == Reg("R7")

    def test_rn_clobbered_in_intermediate_no_fold(self):
        """lo-save register (R7) overwritten before A=B → lookahead aborts."""
        nodes = [
            Assign(0x1000, Reg("A"), Reg("R1")),
            Assign(0x1002, Reg("B"), Const(0x10)),
            Assign(0x1004, RegGroup(("B", "A")), BinOp(Reg("A"), "*", Reg("B"))),
            Assign(0x1006, Reg("R7"), Reg("A")),                              # terminal: lo save
            Assign(0x1008, Reg("R7"), Const(0)),                              # R7 clobbered → abort
            Assign(0x100a, Reg("A"), Reg("B")),
            Assign(0x100c, Reg("R2"), Reg("A")),
            Assign(0x100e, Reg("A"), Reg("R7")),
            Assign(0x1010, Reg("R3"), Reg("A")),
        ]
        result = _match(nodes)
        # Lookahead aborts; terminal is R7=A → R7 = (uint8_t)(R1 * 0x10)
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        node = replacement[0]
        assert isinstance(node, Assign)
        assert node.lhs == Reg("R7")
