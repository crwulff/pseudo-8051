from pseudo8051.ir.hir import Assign
from pseudo8051.ir.expr import Reg
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.passes.patterns.reg_copy_group import RegCopyGroupPattern

_noop_simplify = lambda nodes, reg_map: nodes


class TestRegCopyGroupPattern:
    def _pat(self):
        return RegCopyGroupPattern()

    def test_drop_4_copies(self):
        """R0=R4; R1=R5; R2=R6; R3=R7; where R4..R7 = retval1 → dropped."""
        vinfo = VarInfo("retval1", "uint32_t", ("R4", "R5", "R6", "R7"))
        reg_map = {"R4": vinfo, "R5": vinfo, "R6": vinfo, "R7": vinfo}
        nodes = [
            Assign(0, Reg("R0"), Reg("R4")),
            Assign(2, Reg("R1"), Reg("R5")),
            Assign(4, Reg("R2"), Reg("R6")),
            Assign(6, Reg("R3"), Reg("R7")),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert replacement == []
        assert new_i == 4
        # reg_map updated with destination registers (individual keys only)
        assert reg_map["R0"].name == "retval1"
        assert reg_map["R1"].name == "retval1"
        assert reg_map["R2"].name == "retval1"
        assert reg_map["R3"].name == "retval1"
        assert "R0R1R2R3" not in reg_map

    def test_no_fire_not_starting_from_hi(self):
        """Source does not start from vinfo.regs[0] (high byte) → no match."""
        vinfo = VarInfo("retval1", "uint32_t", ("R4", "R5", "R6", "R7"))
        reg_map = {"R4": vinfo, "R5": vinfo, "R6": vinfo, "R7": vinfo}
        nodes = [
            Assign(0, Reg("R0"), Reg("R7")),   # starts from low byte
            Assign(2, Reg("R1"), Reg("R6")),
            Assign(4, Reg("R2"), Reg("R5")),
            Assign(6, Reg("R3"), Reg("R4")),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_xram_source(self):
        """XRAM-local source VarInfo → no match."""
        vinfo = VarInfo("local1", "uint16_t", ("R6", "R7"),
                        xram_sym="EXT_DC00")
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = [Assign(0, Reg("R0"), Reg("R6")), Assign(2, Reg("R1"), Reg("R7"))]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_wrong_source_sequence(self):
        """Second copy has wrong source register → no match."""
        vinfo = VarInfo("retval1", "uint16_t", ("R6", "R7"))
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = [
            Assign(0, Reg("R0"), Reg("R6")),
            Assign(2, Reg("R1"), Reg("R6")),  # should be R7
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None
