"""
tests/patterns/test_xch_copy.py — Unit tests for XchCopyPattern.
"""

from pseudo8051.ir.hir import Assign, ExprStmt
from pseudo8051.ir.expr import Reg, Const, BinOp, UnaryOp, Call, CROMRef, XRAMRef, RegGroup
from pseudo8051.passes.patterns.xch_copy import XchCopyPattern

_noop = lambda nodes, reg_map: nodes


def _swap(ea, a, b):
    """Build ExprStmt(swap(a, b)) helper."""
    return ExprStmt(ea, Call("swap", [Reg(a), Reg(b)]))


def _inc_dptr(ea):
    return ExprStmt(ea, UnaryOp("++", Reg("DPTR"), post=True))


def _xch_block(base_ea, rlo, rhi):
    """6-instruction XCH swap block for rhi:rlo ↔ DPTR."""
    return [
        _swap(base_ea + 0, "A", rlo),
        _swap(base_ea + 1, "A", "DPL"),
        _swap(base_ea + 2, "A", rlo),
        _swap(base_ea + 3, "A", rhi),
        _swap(base_ea + 4, "A", "DPH"),
        _swap(base_ea + 5, "A", rhi),
    ]


def _movc(ea):
    """A = CROM[A + DPTR]."""
    return Assign(ea, Reg("A"), CROMRef(BinOp(Reg("A"), "+", Reg("DPTR"))))


def _movc_simplified(ea):
    """A = CROM[DPTR]  (already simplified by constant propagation)."""
    return Assign(ea, Reg("A"), CROMRef(Reg("DPTR")))


def _movx(ea):
    """XRAM[DPTR] = A."""
    return Assign(ea, XRAMRef(Reg("DPTR")), Reg("A"))


class TestXchCopyPattern:

    def _pat(self):
        return XchCopyPattern()

    def _full_nodes(self, rlo="R0", rhi="R4", with_clr=True):
        """Build the complete 17- or 16-node XCH copy sequence."""
        nodes = []
        ea = 0x1000
        if with_clr:
            nodes.append(Assign(ea, Reg("A"), Const(0)))   # clr A
            ea += 1
        nodes.append(_movc(ea));    ea += 1                 # movc
        nodes.append(_inc_dptr(ea)); ea += 1                # inc DPTR
        nodes.extend(_xch_block(ea, rlo, rhi)); ea += 6    # first XCH swap
        nodes.append(_movx(ea));    ea += 1                 # movx @DPTR, A
        nodes.append(_inc_dptr(ea)); ea += 1                # inc DPTR
        nodes.extend(_xch_block(ea, rlo, rhi))              # second XCH swap
        return nodes

    def test_full_pattern_with_clr(self):
        """17-node sequence (with clr A) collapses to 3 nodes."""
        nodes = self._full_nodes(rlo="R0", rhi="R4", with_clr=True)
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == len(nodes)
        assert len(repl) == 3

        # XRAM[R4R0] = CROM[DPTR];
        n0 = repl[0]
        assert isinstance(n0, Assign)
        assert isinstance(n0.lhs, XRAMRef)
        assert n0.lhs.inner == RegGroup(("R4", "R0"))
        assert isinstance(n0.rhs, CROMRef)
        assert n0.rhs.inner == Reg("DPTR")
        assert n0.render(0)[0][1] == "XRAM[R4R0] = CROM[DPTR];"

        # DPTR++;
        n1 = repl[1]
        assert isinstance(n1, ExprStmt)
        assert n1.render(0)[0][1] == "DPTR++;"

        # R4R0++;
        n2 = repl[2]
        assert isinstance(n2, ExprStmt)
        assert n2.render(0)[0][1] == "R4R0++;"

    def test_full_pattern_without_clr(self):
        """16-node sequence (no clr A, movc already simplified) collapses to 3 nodes."""
        nodes = []
        ea = 0x1000
        nodes.append(_movc_simplified(ea)); ea += 1
        nodes.append(_inc_dptr(ea));        ea += 1
        nodes.extend(_xch_block(ea, "R0", "R4")); ea += 6
        nodes.append(_movx(ea));            ea += 1
        nodes.append(_inc_dptr(ea));        ea += 1
        nodes.extend(_xch_block(ea, "R0", "R4"))

        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == len(nodes)
        assert len(repl) == 3
        assert repl[0].render(0)[0][1] == "XRAM[R4R0] = CROM[DPTR];"

    def test_different_register_pair(self):
        """Pattern works with any Rhi:Rlo pair, e.g. R2:R1."""
        nodes = self._full_nodes(rlo="R1", rhi="R2", with_clr=True)
        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        repl, _ = result
        assert repl[0].render(0)[0][1] == "XRAM[R2R1] = CROM[DPTR];"
        assert repl[2].render(0)[0][1] == "R2R1++;"

    def test_no_match_on_unrelated_nodes(self):
        """Non-matching nodes return None."""
        nodes = [
            Assign(0, Reg("A"), Reg("R7")),
            Assign(2, XRAMRef(Reg("DPTR")), Reg("A")),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_wrong_swap_order(self):
        """Swaps in wrong order (DPH before DPL) are not matched."""
        nodes = []
        ea = 0x1000
        nodes.append(_movc(ea));     ea += 1
        nodes.append(_inc_dptr(ea)); ea += 1
        # Wrong order: DPH/DPL swapped
        nodes += [
            _swap(ea+0, "A", "R0"),
            _swap(ea+1, "A", "DPH"),   # should be DPL
            _swap(ea+2, "A", "R0"),
            _swap(ea+3, "A", "R4"),
            _swap(ea+4, "A", "DPL"),   # should be DPH
            _swap(ea+5, "A", "R4"),
        ]
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_no_match_mismatched_second_swap(self):
        """Second XCH block uses different registers → no match."""
        nodes = []
        ea = 0x1000
        nodes.append(_movc(ea));     ea += 1
        nodes.append(_inc_dptr(ea)); ea += 1
        nodes.extend(_xch_block(ea, "R0", "R4")); ea += 6
        nodes.append(_movx(ea));     ea += 1
        nodes.append(_inc_dptr(ea)); ea += 1
        # Second swap uses R1/R5 instead of R0/R4
        nodes.extend(_xch_block(ea, "R1", "R5"))
        assert self._pat().match(nodes, 0, {}, _noop) is None

    def test_xram_source(self):
        """XRAM→XRAM: movx A, @DPTR variant produces XRAM[R4R0] = XRAM[DPTR]."""
        nodes = []
        ea = 0x1000
        nodes.append(Assign(ea, Reg("A"), XRAMRef(Reg("DPTR")))); ea += 1  # movx A, @DPTR
        nodes.append(_inc_dptr(ea));                               ea += 1
        nodes.extend(_xch_block(ea, "R0", "R4"));                  ea += 6
        nodes.append(_movx(ea));                                    ea += 1
        nodes.append(_inc_dptr(ea));                               ea += 1
        nodes.extend(_xch_block(ea, "R0", "R4"))

        result = self._pat().match(nodes, 0, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == len(nodes)
        assert len(repl) == 3
        assert repl[0].render(0)[0][1] == "XRAM[R4R0] = XRAM[DPTR];"
        assert repl[1].render(0)[0][1] == "DPTR++;"
        assert repl[2].render(0)[0][1] == "R4R0++;"

    def test_match_at_offset(self):
        """Pattern matches at non-zero index within a longer node list."""
        prefix = [Assign(0, Reg("R7"), Const(5))]
        body = self._full_nodes(with_clr=True)
        nodes = prefix + body
        result = self._pat().match(nodes, 1, {}, _noop)
        assert result is not None
        repl, new_i = result
        assert new_i == 1 + len(body)
        assert len(repl) == 3
