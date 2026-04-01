"""
tests/patterns/test_reg_inc.py — Unit tests for RegPostIncPattern and RegPreIncPattern.
"""

from pseudo8051.ir.hir import Assign, ExprStmt, ReturnStmt, IfGoto
from pseudo8051.ir.expr import Reg, UnaryOp, IRAMRef, XRAMRef, Const, BinOp, Name
from pseudo8051.passes.patterns.reg_inc import RegPostIncPattern, RegPreIncPattern

_noop = lambda nodes, reg_map: nodes


class TestRegPostIncPattern:

    def _pat(self):
        return RegPostIncPattern()

    # ── Basic matches ─────────────────────────────────────────────────────────

    def test_iram_load_post_inc(self):
        """A = IRAM[R1]; R1++; → A = IRAM[R1++];"""
        nodes = [
            Assign(0x100, Reg("A"), IRAMRef(Reg("R1"))),
            ExprStmt(0x101, UnaryOp("++", Reg("R1"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == 2
        assert len(repl) == 1
        assert repl[0].render(0)[0][1] == "A = IRAM[R1++];"

    def test_xram_load_post_inc(self):
        """A = XRAM[R0]; R0++; → A = XRAM[R0++];"""
        nodes = [
            Assign(0x200, Reg("A"), XRAMRef(Reg("R0"))),
            ExprStmt(0x201, UnaryOp("++", Reg("R0"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "A = XRAM[R0++];"

    def test_iram_store_post_inc(self):
        """IRAM[R0] = A; R0++; → IRAM[R0++] = A;"""
        nodes = [
            Assign(0x300, IRAMRef(Reg("R0")), Reg("A")),
            ExprStmt(0x301, UnaryOp("++", Reg("R0"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "IRAM[R0++] = A;"

    def test_xram_store_post_inc(self):
        """XRAM[R1] = A; R1++; → XRAM[R1++] = A;"""
        nodes = [
            Assign(0x400, XRAMRef(Reg("R1")), Reg("A")),
            ExprStmt(0x401, UnaryOp("++", Reg("R1"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "XRAM[R1++] = A;"

    def test_post_dec(self):
        """A = IRAM[R1]; R1--; → A = IRAM[R1--];"""
        nodes = [
            Assign(0x100, Reg("A"), IRAMRef(Reg("R1"))),
            ExprStmt(0x101, UnaryOp("--", Reg("R1"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "A = IRAM[R1--];"

    def test_dptr_post_inc(self):
        """A = XRAM[DPTR]; DPTR++; → A = XRAM[DPTR++]; (DPTR is now included)"""
        nodes = [
            Assign(0x100, Reg("A"), XRAMRef(Reg("DPTR"))),
            ExprStmt(0x101, UnaryOp("++", Reg("DPTR"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "A = XRAM[DPTR++];"

    def test_return_stmt_post_inc(self):
        """return R1; R1++; → return R1++;"""
        nodes = [
            ReturnStmt(0x100, Reg("R1")),
            ExprStmt(0x101, UnaryOp("++", Reg("R1"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "return R1++;"

    def test_ifgoto_post_inc(self):
        """if (R1 != 0) goto L; R1++; → if (R1++ != 0) goto L;"""
        nodes = [
            IfGoto(0x100, BinOp(Reg("R1"), "!=", Const(0)), "L"),
            ExprStmt(0x101, UnaryOp("++", Reg("R1"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        rendered = result[0][0].render(0)[0][1]
        assert "R1++" in rendered

    def test_ea_from_first_node(self):
        """Output node carries ea from n0, not the inc."""
        nodes = [
            Assign(0xABC, Reg("A"), IRAMRef(Reg("R2"))),
            ExprStmt(0xABD, UnaryOp("++", Reg("R2"), post=True)),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].ea == 0xABC

    def test_match_at_offset(self):
        """Pattern works at non-zero index."""
        prefix = [Assign(0, Reg("R7"), Reg("A"))]
        body = [
            Assign(0x10, Reg("A"), IRAMRef(Reg("R1"))),
            ExprStmt(0x11, UnaryOp("++", Reg("R1"), post=True)),
        ]
        nodes = prefix + body
        result = self._pat().match(nodes, 1, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == 3
        assert repl[0].render(0)[0][1] == "A = IRAM[R1++];"

    # ── No-match cases ────────────────────────────────────────────────────────

    def test_no_match_pre_inc(self):
        """Pre-increment (post=False) ExprStmt is not the post-increment idiom."""
        nodes = [
            Assign(0, Reg("A"), IRAMRef(Reg("R1"))),
            ExprStmt(1, UnaryOp("++", Reg("R1"), post=False)),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_different_reg(self):
        """Inc targets a different register than used in n0."""
        nodes = [
            Assign(0, Reg("A"), IRAMRef(Reg("R1"))),
            ExprStmt(1, UnaryOp("++", Reg("R0"), post=True)),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_two_uses(self):
        """Reg appears twice in n0 — cannot safely fold (which occurrence to pick?)."""
        nodes = [
            Assign(0, IRAMRef(Reg("R1")), Reg("R1")),  # R1 on both sides → 2 reads (LHS inner + RHS)
            ExprStmt(1, UnaryOp("++", Reg("R1"), post=True)),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_zero_uses(self):
        """Reg does not appear in n0 at all."""
        nodes = [
            Assign(0, Reg("A"), Const(5)),
            ExprStmt(1, UnaryOp("++", Reg("R1"), post=True)),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_single_node(self):
        """Only one node — cannot form the pair."""
        nodes = [Assign(0, Reg("A"), IRAMRef(Reg("R1")))]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_n0_is_inc(self):
        """n0 is itself an inc node — guard prevents chaining."""
        nodes = [
            ExprStmt(0, UnaryOp("++", Reg("R0"), post=True)),
            ExprStmt(1, UnaryOp("++", Reg("R0"), post=True)),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_not_exprstmt(self):
        """n1 is not an ExprStmt — no fold."""
        nodes = [
            Assign(0, Reg("A"), IRAMRef(Reg("R1"))),
            Assign(1, Reg("R1"), Const(0)),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None


class TestRegPreIncPattern:

    def _pat(self):
        return RegPreIncPattern()

    # ── Basic matches ─────────────────────────────────────────────────────────

    def test_iram_pre_inc(self):
        """R1++; A = IRAM[R1]; → A = IRAM[++R1];"""
        nodes = [
            ExprStmt(0x100, UnaryOp("++", Reg("R1"), post=True)),
            Assign(0x101, Reg("A"), IRAMRef(Reg("R1"))),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == 2
        assert len(repl) == 1
        assert repl[0].render(0)[0][1] == "A = IRAM[++R1];"

    def test_xram_pre_inc(self):
        """R0++; A = XRAM[R0]; → A = XRAM[++R0];"""
        nodes = [
            ExprStmt(0x100, UnaryOp("++", Reg("R0"), post=True)),
            Assign(0x101, Reg("A"), XRAMRef(Reg("R0"))),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "A = XRAM[++R0];"

    def test_iram_store_pre_inc(self):
        """R0++; IRAM[R0] = A; → IRAM[++R0] = A;"""
        nodes = [
            ExprStmt(0x100, UnaryOp("++", Reg("R0"), post=True)),
            Assign(0x101, IRAMRef(Reg("R0")), Reg("A")),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "IRAM[++R0] = A;"

    def test_pre_dec(self):
        """R1--; A = IRAM[R1]; → A = IRAM[--R1];"""
        nodes = [
            ExprStmt(0x100, UnaryOp("--", Reg("R1"), post=True)),
            Assign(0x101, Reg("A"), IRAMRef(Reg("R1"))),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "A = IRAM[--R1];"

    def test_dptr_pre_inc(self):
        """DPTR++; A = XRAM[DPTR]; → A = XRAM[++DPTR];"""
        nodes = [
            ExprStmt(0x100, UnaryOp("++", Reg("DPTR"), post=True)),
            Assign(0x101, Reg("A"), XRAMRef(Reg("DPTR"))),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].render(0)[0][1] == "A = XRAM[++DPTR];"

    def test_ea_from_n1(self):
        """Output node carries ea from n1 (the use node), not the inc."""
        nodes = [
            ExprStmt(0xABC, UnaryOp("++", Reg("R2"), post=True)),
            Assign(0xABD, Reg("A"), IRAMRef(Reg("R2"))),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].ea == 0xABD

    def test_match_at_offset(self):
        """Pattern works at non-zero index."""
        prefix = [Assign(0, Reg("R7"), Reg("A"))]
        body = [
            ExprStmt(0x10, UnaryOp("++", Reg("R1"), post=True)),
            Assign(0x11, Reg("A"), IRAMRef(Reg("R1"))),
        ]
        nodes = prefix + body
        result = self._pat().match(nodes, 1, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == 3
        assert repl[0].render(0)[0][1] == "A = IRAM[++R1];"

    # ── No-match cases ────────────────────────────────────────────────────────

    def test_no_match_n0_not_inc(self):
        """n0 is not an ExprStmt(Rn++) — no fold."""
        nodes = [
            Assign(0, Reg("A"), IRAMRef(Reg("R1"))),
            Assign(1, Reg("R7"), Reg("A")),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_n0_pre_form(self):
        """ExprStmt(++R1) with post=False is not matched (only post=True ExprStmts)."""
        nodes = [
            ExprStmt(0, UnaryOp("++", Reg("R1"), post=False)),
            Assign(1, Reg("A"), IRAMRef(Reg("R1"))),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_different_reg(self):
        """Inc targets a different register than used in n1."""
        nodes = [
            ExprStmt(0, UnaryOp("++", Reg("R0"), post=True)),
            Assign(1, Reg("A"), IRAMRef(Reg("R1"))),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_two_uses(self):
        """Reg appears twice in n1 — cannot safely fold."""
        nodes = [
            ExprStmt(0, UnaryOp("++", Reg("R1"), post=True)),
            Assign(1, IRAMRef(Reg("R1")), Reg("R1")),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_single_node(self):
        """Only one node — cannot form the pair."""
        nodes = [ExprStmt(0, UnaryOp("++", Reg("R1"), post=True))]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_n1_is_inc(self):
        """n1 is itself an inc node — guard prevents chaining."""
        nodes = [
            ExprStmt(0, UnaryOp("++", Reg("R1"), post=True)),
            ExprStmt(1, UnaryOp("++", Reg("R1"), post=True)),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None
