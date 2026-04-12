"""
tests/patterns/test_rol_switch.py — Unit tests for RolSwitchPattern.

Verifies collapsing of the 8051 indirect-jump preamble:

    A = rol8(A); A = rol8(A); A |= DPL; A = rol8(A); switch (A >> 1) {
    → switch ((A << 2) | DPL) {
"""

from pseudo8051.ir.hir import Assign, CompoundAssign, SwitchNode
from pseudo8051.ir.expr import Reg, Const, BinOp
from pseudo8051.passes.patterns.rol_switch import RolSwitchPattern, _is_rol_a

_noop = lambda nodes, reg_map: nodes


def _rol_a(ea: int) -> Assign:
    a = Reg("A")
    return Assign(ea, a, BinOp(BinOp(a, "<<", Const(1)), "|", BinOp(a, ">>", Const(7))))


def _sw(ea: int, k: int, cases=None) -> SwitchNode:
    """SwitchNode with subject A >> k (or just A when k=0)."""
    if cases is None:
        cases = [([0], "label_0"), ([1], "label_1")]
    if k == 0:
        subject = Reg("A")
    else:
        subject = BinOp(Reg("A"), ">>", Const(k))
    return SwitchNode(ea, subject, cases)


class TestIsRolA:
    def test_recognises_rol8_a(self):
        assert _is_rol_a(_rol_a(0))

    def test_rejects_ror(self):
        # right-rotate: (A >> 1) | (A << 7) — different shifts
        a = Reg("A")
        node = Assign(0, a, BinOp(BinOp(a, ">>", Const(1)), "|", BinOp(a, "<<", Const(7))))
        assert not _is_rol_a(node)

    def test_rejects_wrong_shift_amount(self):
        # (A << 2) | (A >> 6) — not a single-bit rotate
        a = Reg("A")
        node = Assign(0, a, BinOp(BinOp(a, "<<", Const(2)), "|", BinOp(a, ">>", Const(6))))
        assert not _is_rol_a(node)

    def test_rejects_non_assign(self):
        from pseudo8051.ir.hir import ExprStmt
        from pseudo8051.ir.expr import UnaryOp
        node = ExprStmt(0, UnaryOp("++", Reg("A"), post=True))
        assert not _is_rol_a(node)


class TestRolSwitchPattern:

    def _pat(self):
        return RolSwitchPattern()

    # ── User's exact example ──────────────────────────────────────────────────

    def test_two_rols_compound_or_one_step(self):
        """rl A; rl A; orl A, DPL; rl A; switch(A >> 1) → switch((A << 2) | DPL)"""
        nodes = [
            _rol_a(0x100),
            _rol_a(0x101),
            CompoundAssign(0x102, Reg("A"), "|=", Reg("DPL")),
            _rol_a(0x103),
            _sw(0x104, k=1),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == 5
        assert len(repl) == 1
        sw = repl[0]
        assert isinstance(sw, SwitchNode)
        assert sw.subject.render() == "A << 2 | DPL"

    # ── Prefix rols only (no compound) ───────────────────────────────────────

    def test_three_rols_step1(self):
        """rl rl rl switch(A>>1) → switch(A<<2)  (3 rols, k=1 ⇒ 1 step, 2 prefix)"""
        nodes = [_rol_a(0), _rol_a(1), _rol_a(2), _sw(3, k=1)]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        sw = result[0][0]
        assert sw.subject.render() == "A << 2"

    def test_two_rols_step2(self):
        """rl rl switch(A>>2) → switch(A)  (2 rols all step-size for 4-byte table)"""
        nodes = [_rol_a(0), _rol_a(1), _sw(2, k=2)]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        sw = result[0][0]
        assert sw.subject.render() == "A"

    def test_one_rol_step1_trivial(self):
        """rl switch(A>>1) → switch(A)  (single rol cancelled by >>1)"""
        nodes = [_rol_a(0), _sw(1, k=1)]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        sw = result[0][0]
        assert sw.subject.render() == "A"

    def test_four_rols_step2(self):
        """rl rl rl rl switch(A>>2) → switch(A<<2)  (4 rols, k=2)"""
        nodes = [_rol_a(0), _rol_a(1), _rol_a(2), _rol_a(3), _sw(4, k=2)]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        sw = result[0][0]
        assert sw.subject.render() == "A << 2"

    # ── With compound assigns ─────────────────────────────────────────────────

    def test_one_prefix_and_compound(self):
        """rl A; A &= 0x7; rl A; switch(A>>1) → switch((A<<1) & 7)"""
        nodes = [
            _rol_a(0),
            CompoundAssign(1, Reg("A"), "&=", Const(7)),
            _rol_a(2),
            _sw(3, k=1),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        sw = result[0][0]
        assert sw.subject.render() == "A << 1 & 7"

    def test_multiple_compounds(self):
        """rl A; A |= R1; A &= 0x7; rl A; switch(A>>1)"""
        nodes = [
            _rol_a(0),
            CompoundAssign(1, Reg("A"), "|=", Reg("R1")),
            CompoundAssign(2, Reg("A"), "&=", Const(7)),
            _rol_a(3),
            _sw(4, k=1),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        sw = result[0][0]
        # A << 1 | R1 & 7  (rendered with precedence)
        rendered = sw.subject.render()
        assert "A << 1" in rendered and "R1" in rendered and "7" in rendered

    def test_ea_from_switch(self):
        """Output SwitchNode preserves the switch's ea."""
        nodes = [_rol_a(0x100), _rol_a(0x101), _sw(0xDEAD, k=1)]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].ea == 0xDEAD

    def test_cases_preserved(self):
        """Switch cases are unchanged after simplification."""
        my_cases = [([2], "lbl_a"), ([4], "lbl_b")]
        nodes = [_rol_a(0), _sw(1, k=1, cases=my_cases)]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        assert result[0][0].cases is my_cases

    def test_match_at_offset(self):
        """Pattern works at non-zero index."""
        prefix = [Assign(0, Reg("R7"), Reg("A"))]
        body = [_rol_a(1), _rol_a(2), _sw(3, k=1)]
        nodes = prefix + body
        result = self._pat().match(nodes, 1, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == 4

    def test_extra_after_rols(self):
        """rl A; A |= DPL; rl rl switch(A>>2) — n_after=2 ≥ k=2, n_prefix=1"""
        nodes = [
            _rol_a(0),
            CompoundAssign(1, Reg("A"), "|=", Reg("DPL")),
            _rol_a(2),
            _rol_a(3),
            _sw(4, k=2),
        ]
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        sw = result[0][0]
        # n_before=1, n_after=2, k=2 → prefix=1+(2-2)=1, step=2
        assert "A << 1" in sw.subject.render()
        assert "DPL" in sw.subject.render()

    # ── No-match cases ────────────────────────────────────────────────────────

    def test_no_match_not_starting_with_rol(self):
        """Pattern must start with A = rol8(A)."""
        nodes = [
            CompoundAssign(0, Reg("A"), "|=", Reg("DPL")),
            _rol_a(1),
            _sw(2, k=1),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_not_enough_rols_to_cancel(self):
        """rl switch(A>>2) — only 1 rol for k=2: can't cancel."""
        nodes = [_rol_a(0), _sw(1, k=2)]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_terminal_not_switch(self):
        """A series of rols with no SwitchNode terminal."""
        from pseudo8051.ir.hir import Assign
        nodes = [_rol_a(0), _rol_a(1), Assign(2, Reg("R0"), Reg("A"))]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_switch_subject_not_a(self):
        """Switch subject is R0 >> 1 — not an A-expression."""
        sw = SwitchNode(0, BinOp(Reg("R0"), ">>", Const(1)), [])
        nodes = [_rol_a(0), _rol_a(1), sw]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_compounds_block_borrow(self):
        """rl; compound; (no more rols); switch(A>>1) — compounds separate prefix from step."""
        nodes = [
            _rol_a(0),
            CompoundAssign(1, Reg("A"), "|=", Reg("DPL")),
            _sw(2, k=1),
        ]
        # n_before=1, compounds=1, n_after=0, k=1
        # n_after(0) < k(1) AND compounds non-empty → no match
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_switch_shift_not_const(self):
        """switch(A >> R0) — shift amount not a Const."""
        sw = SwitchNode(0, BinOp(Reg("A"), ">>", Reg("R0")), [])
        nodes = [_rol_a(0), sw]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_single_node(self):
        """Only one node — can't form the pair with a switch."""
        nodes = [_rol_a(0)]
        assert self._pat().match(nodes, 0, {}, _noop) is None
