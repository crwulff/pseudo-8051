from pseudo8051.passes.typesimplify._post import _consolidate_xram_local_loads
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.ir.hir import Assign
from pseudo8051.ir.expr import Reg, Name


def _dest_reg_map():
    parent = VarInfo("_dest", "uint16_t", (), xram_sym="EXT_DC68")
    hi = VarInfo("_dest.hi", "uint8_t", (), xram_sym="EXT_DC68", is_byte_field=True)
    lo = VarInfo("_dest.lo", "uint8_t", (), xram_sym="EXT_DC69", is_byte_field=True)
    return {"_dest": parent, "_byte_EXT_DC68": hi, "_byte_EXT_DC69": lo}


class TestConsolidateXramLocalLoads:

    def test_r4dpl_then_dph_collapses_to_dptr(self):
        """R4=_dest.hi; DPL=_dest.lo; DPH=R4; → DPTR=_dest;"""
        nodes = [
            Assign(0x100, Reg("R4"),  Name("_dest.hi")),
            Assign(0x102, Reg("DPL"), Name("_dest.lo")),
            Assign(0x104, Reg("DPH"), Reg("R4")),
        ]
        result = _consolidate_xram_local_loads(nodes, _dest_reg_map())
        assert len(result) == 1
        assert result[0].lhs.render() == "DPTR"
        assert result[0].rhs.render() == "_dest"

    def test_r4dpl_no_trailing_dph_stays_reggroup(self):
        """Without trailing DPH=R4, result is RegGroup(R4,DPL)=_dest."""
        nodes = [
            Assign(0x100, Reg("R4"),  Name("_dest.hi")),
            Assign(0x102, Reg("DPL"), Name("_dest.lo")),
        ]
        result = _consolidate_xram_local_loads(nodes, _dest_reg_map())
        assert len(result) == 1
        assert result[0].lhs.render() == "R4DPL"

    def test_wrong_reg_for_dph_no_collapse(self):
        """DPH=R5 (not the hi-byte reg R4) → RegGroup kept, DPH stmt left over."""
        nodes = [
            Assign(0x100, Reg("R4"),  Name("_dest.hi")),
            Assign(0x102, Reg("DPL"), Name("_dest.lo")),
            Assign(0x104, Reg("DPH"), Reg("R5")),
        ]
        result = _consolidate_xram_local_loads(nodes, _dest_reg_map())
        assert len(result) == 2
        assert result[0].lhs.render() == "R4DPL"
        assert result[1].lhs.render() == "DPH"

    def test_non_dpl_lo_byte_unaffected(self):
        """Normal R4R5 pair (lo byte is R5, not DPL) → standard RegGroup, no DPTR."""
        parent = VarInfo("src", "uint16_t", (), xram_sym="EXT_DC70")
        hi = VarInfo("src.hi", "uint8_t", (), xram_sym="EXT_DC70", is_byte_field=True)
        lo = VarInfo("src.lo", "uint8_t", (), xram_sym="EXT_DC71", is_byte_field=True)
        reg_map = {"src": parent, "_byte_EXT_DC70": hi, "_byte_EXT_DC71": lo}
        nodes = [
            Assign(0x100, Reg("R4"), Name("src.hi")),
            Assign(0x102, Reg("R5"), Name("src.lo")),
        ]
        result = _consolidate_xram_local_loads(nodes, reg_map)
        assert len(result) == 1
        assert result[0].lhs.render() == "R4R5"
