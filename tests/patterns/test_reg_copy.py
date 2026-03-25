from pseudo8051.ir.hir import Statement
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.passes.patterns.reg_copy_group import RegCopyGroupPattern

_noop_simplify = lambda nodes, reg_map: nodes


class TestRegCopyGroupPattern:
    def _pat(self):
        return RegCopyGroupPattern()

    def test_drop_4_copies(self):
        """R0=R4; R1=R5; R2=R6; R3=R7; where R4R5R6R7 = retval1 → dropped."""
        vinfo = VarInfo("retval1", "uint32_t", ("R4", "R5", "R6", "R7"))
        reg_map = {
            "R4R5R6R7": vinfo, "R4": vinfo, "R5": vinfo,
            "R6": vinfo, "R7": vinfo,
        }
        nodes = [
            Statement(0, "R0 = R4;"),
            Statement(2, "R1 = R5;"),
            Statement(4, "R2 = R6;"),
            Statement(6, "R3 = R7;"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert replacement == []
        assert new_i == 4
        # reg_map updated with destination registers
        assert "R0R1R2R3" in reg_map
        assert reg_map["R0R1R2R3"].name == "retval1"
        assert reg_map["R0"].name == "retval1"

    def test_no_fire_not_starting_from_hi(self):
        """Source does not start from vinfo.regs[0] (high byte) → no match."""
        vinfo = VarInfo("retval1", "uint32_t", ("R4", "R5", "R6", "R7"))
        reg_map = {"R4R5R6R7": vinfo, "R4": vinfo, "R5": vinfo,
                   "R6": vinfo, "R7": vinfo}
        nodes = [
            Statement(0, "R0 = R7;"),  # starts from low byte
            Statement(2, "R1 = R6;"),
            Statement(4, "R2 = R5;"),
            Statement(6, "R3 = R4;"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_xram_source(self):
        """XRAM-local source VarInfo → no match."""
        vinfo = VarInfo("local1", "uint16_t", ("R6", "R7"),
                        xram_sym="EXT_DC00")
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        nodes = [Statement(0, "R0 = R6;"), Statement(2, "R1 = R7;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_wrong_source_sequence(self):
        """Second copy has wrong source register → no match."""
        vinfo = VarInfo("retval1", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        nodes = [
            Statement(0, "R0 = R6;"),
            Statement(2, "R1 = R6;"),  # should be R7
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None
