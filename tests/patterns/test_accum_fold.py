"""
tests/patterns/test_accum_fold.py — AccumFoldPattern edge cases:
  - A += A → A * 2 normalization (issue 2.2)
  - MUL AB folding (issue 2.1)
  - combined chains
"""

from pseudo8051.ir.hir import Assign, CompoundAssign, ExprStmt, IfGoto, ReturnStmt
from pseudo8051.ir.expr import Reg, Regs, Const, BinOp, UnaryOp, XRAMRef, Name, RegGroup, Cast


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
        assert isinstance(node.lhs, Regs) and not node.lhs.is_single
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
        assert isinstance(node.lhs, Regs) and not node.lhs.is_single
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
        assert isinstance(node.lhs, Regs) and not node.lhs.is_single
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


class TestCjneCarryPattern:
    """Tests for the CJNE no-op + JNC/JC combined terminal in AccumFoldPattern."""

    def _match(self, nodes):
        from pseudo8051.passes.patterns.accum_fold import AccumFoldPattern
        return AccumFoldPattern().match(nodes, 0, {}, lambda ns, rm: ns)

    def _cjne_ifgoto(self, ea, limit, label):
        return IfGoto(ea, BinOp(Reg("A"), "!=", Const(limit)), label)

    def _carry_ifgoto_jnc(self, ea, label):
        return IfGoto(ea, UnaryOp("!", Reg("C"), post=False), label)

    def _carry_ifgoto_jc(self, ea, label):
        return IfGoto(ea, Reg("C"), label)

    # ── Basic cases ───────────────────────────────────────────────────────────

    def test_basic_jnc_no_downstream_a(self):
        """A=R7; A+=2; if(A!=4) L; L: if(!C) target → if(R7+2 >= 4) target."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0x100, Reg("A"), Reg("R7")),
            CompoundAssign(0x102, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0x104, 4, "L"),
            Label(0x106, "L"),
            self._carry_ifgoto_jnc(0x106, "target"),
        ]
        result = self._match(nodes)
        assert result is not None
        repl, new_i = result
        assert new_i == 5
        assert len(repl) == 1
        node = repl[0]
        assert isinstance(node, IfGoto)
        rendered = node.cond.render()
        assert "R7" in rendered
        assert ">=" in rendered
        assert "4" in rendered
        assert node.label == "target"

    def test_basic_jc_lt(self):
        """A=R7; A+=2; if(A!=4) L; L: if(C) target → if(R7+2 < 4) target."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0x100, Reg("A"), Reg("R7")),
            CompoundAssign(0x102, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0x104, 4, "L"),
            Label(0x106, "L"),
            self._carry_ifgoto_jc(0x106, "target"),
        ]
        result = self._match(nodes)
        assert result is not None
        repl, new_i = result
        assert new_i == 5
        node = repl[0]
        assert isinstance(node, IfGoto)
        rendered = node.cond.render()
        assert "<" in rendered
        assert "R7" in rendered
        assert node.label == "target"

    # ── ea preserved ─────────────────────────────────────────────────────────

    def test_ea_from_a_start(self):
        """Output IfGoto.ea matches A=... start node's ea."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0xDEAD, Reg("A"), Reg("R7")),
            CompoundAssign(0xDEAE, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0xDEB0, 4, "L"),
            Label(0xDEB2, "L"),
            self._carry_ifgoto_jnc(0xDEB2, "tgt"),
        ]
        result = self._match(nodes)
        assert result is not None
        node = result[0][0]
        assert node.ea == 0xDEAD

    # ── Re-emit A when used downstream ───────────────────────────────────────

    def test_a_reemitted_when_downstream_uses_a(self):
        """When a later node reads A, Assign(A, expr) is prepended."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0x100, Reg("A"), Reg("R7")),
            CompoundAssign(0x102, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0x104, 4, "L"),
            Label(0x106, "L"),
            self._carry_ifgoto_jnc(0x106, "target"),
            Assign(0x108, Reg("DPL"), Reg("A")),   # downstream uses A
        ]
        result = self._match(nodes)
        assert result is not None
        repl, new_i = result
        assert new_i == 5
        assert len(repl) == 2
        assert isinstance(repl[0], Assign)
        assert repl[0].lhs == Reg("A")
        assert "R7" in repl[0].rhs.render()
        assert isinstance(repl[1], IfGoto)
        assert ">=" in repl[1].cond.render()

    def test_a_not_reemitted_when_not_used_downstream(self):
        """No downstream A use → no extra Assign node."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0x100, Reg("A"), Reg("R7")),
            CompoundAssign(0x102, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0x104, 4, "L"),
            Label(0x106, "L"),
            self._carry_ifgoto_jnc(0x106, "target"),
            Assign(0x108, Reg("DPL"), Reg("R5")),   # does NOT read A
        ]
        result = self._match(nodes)
        assert result is not None
        repl, _ = result
        assert len(repl) == 1
        assert isinstance(repl[0], IfGoto)

    # ── No-match cases ────────────────────────────────────────────────────────

    def test_no_match_label_mismatch(self):
        """Label name differs from CJNE target → falls through to normal IfGoto."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0x100, Reg("A"), Reg("R7")),
            CompoundAssign(0x102, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0x104, 4, "L_cjne"),
            Label(0x106, "L_other"),               # mismatch
            self._carry_ifgoto_jnc(0x106, "target"),
        ]
        result = self._match(nodes)
        # Falls through to normal IfGoto: condition has "!=" not ">="
        if result is not None:
            for node in result[0]:
                if isinstance(node, IfGoto):
                    assert ">=" not in node.cond.render()

    def test_no_match_no_label_after_cjne(self):
        """Missing Label node → no combined match."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0x100, Reg("A"), Reg("R7")),
            CompoundAssign(0x102, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0x104, 4, "L"),
            # Label missing — carry IfGoto immediately follows
            self._carry_ifgoto_jnc(0x106, "target"),
        ]
        result = self._match(nodes)
        if result is not None:
            for node in result[0]:
                if isinstance(node, IfGoto):
                    assert ">=" not in node.cond.render()

    def test_no_match_carry_node_missing(self):
        """Label present but next node is not a carry IfGoto."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0x100, Reg("A"), Reg("R7")),
            CompoundAssign(0x102, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0x104, 4, "L"),
            Label(0x106, "L"),
            Assign(0x106, Reg("DPL"), Reg("R0")),
        ]
        result = self._match(nodes)
        if result is not None:
            for node in result[0]:
                if isinstance(node, IfGoto):
                    assert ">=" not in node.cond.render()

    def test_no_match_carry_cond_not_c(self):
        """IfGoto after label uses non-carry condition → no combined match."""
        from pseudo8051.ir.hir import Label
        nodes = [
            Assign(0x100, Reg("A"), Reg("R7")),
            CompoundAssign(0x102, Reg("A"), "+=", Const(2)),
            self._cjne_ifgoto(0x104, 4, "L"),
            Label(0x106, "L"),
            IfGoto(0x106, BinOp(Reg("R0"), "==", Const(0)), "target"),
        ]
        result = self._match(nodes)
        if result is not None:
            for node in result[0]:
                if isinstance(node, IfGoto):
                    assert ">=" not in node.cond.render()
