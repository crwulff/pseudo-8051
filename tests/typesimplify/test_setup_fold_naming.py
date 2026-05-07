"""
tests/typesimplify/test_setup_fold_naming.py

Tests for _fold_call_arg_pairs naming correctness when the hi-byte source
is an XRAM load that does not match the register pair's declared variable.
"""

from pseudo8051.passes.typesimplify._setup_fold import _fold_call_arg_pairs
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.ir.hir import Assign, TypedAssign
from pseudo8051.ir.expr import Reg, Regs, Name, XRAMRef


EA = 0x1000


def _reg(name: str):
    return Reg(name)


class TestFoldCallArgPairsNaming:

    def _run(self, nodes, reg_map):
        return _fold_call_arg_pairs(list(nodes), reg_map)

    def test_xram_source_unknown_suppresses_register_pair_name(self):
        """R6 = XRAM[EXT_DC43], R7 = XRAM[EXT_DC44] should NOT produce
        TypedAssign named 'osdAddr' when osdAddr lives at EXT_DC41/DC42."""
        vi_osd = VarInfo("osdAddr", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vi_osd, "R7": vi_osd, "R6R7": vi_osd}
        nodes = [
            Assign(EA,       _reg("R6"), XRAMRef(Name("EXT_DC43"))),
            Assign(EA + 1,   _reg("R7"), XRAMRef(Name("EXT_DC44"))),
        ]
        result = self._run(nodes, reg_map)
        assert len(result) == 1
        node = result[0]
        # Must NOT be a TypedAssign — name should be suppressed
        assert not isinstance(node, TypedAssign), (
            f"Expected plain Assign, got TypedAssign with lhs={node.lhs!r}"
        )

    def test_xram_source_known_uses_xram_parent_name(self):
        """R6 = XRAM[EXT_DC43], R7 = XRAM[EXT_DC44] produces TypedAssign
        named 'newVar' when reg_map has a parent VarInfo at EXT_DC43."""
        vi_osd = VarInfo("osdAddr", "int16_t", ("R6", "R7"))
        vi_new = VarInfo("newVar", "int16_t", (), xram_sym="EXT_DC43",
                         xram_addr=0xdc43)
        reg_map = {
            "R6": vi_osd, "R7": vi_osd, "R6R7": vi_osd,
            "EXT_DC43": vi_new,
        }
        nodes = [
            Assign(EA,       _reg("R6"), XRAMRef(Name("EXT_DC43"))),
            Assign(EA + 1,   _reg("R7"), XRAMRef(Name("EXT_DC44"))),
        ]
        result = self._run(nodes, reg_map)
        assert len(result) == 1
        node = result[0]
        assert isinstance(node, TypedAssign)
        assert node.lhs.render() == "newVar"

    def test_non_xram_source_keeps_register_pair_name(self):
        """R6 = Name('foo'), R7 = Name('bar') (not XRAM) should keep the
        register pair's declared name 'osdAddr'."""
        vi_osd = VarInfo("osdAddr", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vi_osd, "R7": vi_osd, "R6R7": vi_osd}
        nodes = [
            Assign(EA,       _reg("R6"), Name("foo")),
            Assign(EA + 1,   _reg("R7"), Name("bar")),
        ]
        result = self._run(nodes, reg_map)
        assert len(result) == 1
        node = result[0]
        assert isinstance(node, TypedAssign)
        assert node.lhs.render() == "osdAddr"

    def test_interleaved_write_to_source_reg_falls_back_to_pair_reg(self):
        """R7=R3; [R3=0 (interleaved)]; R6=0 should produce Cast(int16_t, R7)
        not Cast(int16_t, R3), because R3 is overwritten between the two
        byte-assign nodes and is therefore stale at the call site.

        The regression: _fold_call_arg_pairs used to propagate Reg('R3') as
        the lo-byte expression, which _subst_from_reg_exprs then replaced with
        Const(0) (because the annotation at the call site says R3=0), yielding
        a wrong (int16_t)0 argument instead of (int16_t)arg3."""
        vi = VarInfo("a", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vi, "R7": vi}
        nodes = [
            Assign(EA,       _reg("R7"), _reg("R3")),   # R7 = R3 (copy arg3)
            Assign(EA + 1,   _reg("R3"), Regs(("A",))), # R3 = A (interleaved: clobbers R3)
            Assign(EA + 2,   _reg("R6"), Regs(("A",))), # R6 = 0 (hi byte)
        ]
        result = self._run(nodes, reg_map)
        # The combined node for R6R7 should use R7, not R3, as the lo-byte source.
        # The R3=A interleaved node is not consumed, so two nodes remain.
        combined_node = next(
            (n for n in result if hasattr(n, "lhs") and "R6" in getattr(n.lhs, "names", ())),
            None,
        )
        assert combined_node is not None, f"No R6R7 combined node in result: {result!r}"
        rendered = combined_node.rhs.render()
        assert "R7" in rendered, f"Expected R7 in combined rhs, got: {rendered!r}"
        assert "R3" not in rendered, f"R3 should not appear in combined rhs, got: {rendered!r}"
