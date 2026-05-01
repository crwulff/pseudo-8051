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
