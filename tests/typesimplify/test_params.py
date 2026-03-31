from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param
from pseudo8051.ir.hir import Assign, ReturnStmt
from pseudo8051.ir.expr import Reg, XRAMRef, Name

from ..helpers import make_single_block_func


def _rendered(func):
    return [t for n in func.hir for _, t in n.render()]


class TestSingleRegParam:
    def test_basic_accum_relay_with_param(self):
        """A = R7; XRAM[X] = A; with proto count→R7 → XRAM[X] = count;"""
        PROTOTYPES["f"] = FuncProto(
            return_type="void",
            params=[Param("count", "uint8_t", ("R7",))],
        )
        func = make_single_block_func("f", [
            Assign(0x1000, Reg("A"), Reg("R7")),
            Assign(0x1002, XRAMRef(Name("X")), Reg("A")),
        ])
        TypeAwareSimplifier().run(func)
        texts = _rendered(func)
        assert "XRAM[X] = count;" in texts

    def test_no_proto_structural_patterns_run(self):
        """Without a prototype, structural patterns (AccumRelayPattern) still fire."""
        func = make_single_block_func("unknown_fn", [
            Assign(0x1000, Reg("A"), Reg("R7")),
            Assign(0x1002, XRAMRef(Name("X")), Reg("A")),
        ])
        TypeAwareSimplifier().run(func)
        texts = _rendered(func)
        assert "XRAM[X] = R7;" in texts
        assert "A = R7;" not in texts
        assert "XRAM[X] = A;" not in texts

    def test_return_statement_preserved(self):
        """return; statement passes through unchanged."""
        PROTOTYPES["g"] = FuncProto(
            return_type="void",
            params=[Param("val", "uint8_t", ("R7",))],
        )
        func = make_single_block_func("g", [
            Assign(0x1000, Reg("A"), Reg("R7")),
            Assign(0x1002, XRAMRef(Name("X")), Reg("A")),
            ReturnStmt(0x1004, None),
        ])
        TypeAwareSimplifier().run(func)
        texts = _rendered(func)
        assert "return;" in texts


class TestUsercallParamSubst:
    def test_three_relay_pairs(self):
        """Three A-relay pairs with params in R7, R5, R3 → all substituted."""
        PROTOTYPES["h"] = FuncProto(
            return_type="void",
            params=[
                Param("H", "uint8_t", ("R7",)),
                Param("M", "uint8_t", ("R5",)),
                Param("L", "uint8_t", ("R3",)),
            ],
        )
        func = make_single_block_func("h", [
            Assign(0x1000, Reg("A"), Reg("R7")),
            Assign(0x1002, XRAMRef(Name("X1")), Reg("A")),
            Assign(0x1004, Reg("A"), Reg("R5")),
            Assign(0x1006, XRAMRef(Name("X2")), Reg("A")),
            Assign(0x1008, Reg("A"), Reg("R3")),
            Assign(0x100a, XRAMRef(Name("X3")), Reg("A")),
        ])
        TypeAwareSimplifier().run(func)
        texts = _rendered(func)
        assert "XRAM[X1] = H;" in texts
        assert "XRAM[X2] = M;" in texts
        assert "XRAM[X3] = L;" in texts

    def test_pair_param(self):
        """16-bit param in R6R7: pair substituted in expression."""
        PROTOTYPES["p"] = FuncProto(
            return_type="void",
            params=[Param("val", "uint16_t", ("R6", "R7"))],
        )
        func = make_single_block_func("p", ["XRAM[X] = R6R7;"])
        TypeAwareSimplifier().run(func)
        texts = [t for n in func.hir for _, t in n.render()]
        assert "XRAM[X] = val;" in texts


class TestParamKillOnOverwrite:
    """Param-register substitution must stop after the register is overwritten."""

    def _rendered(self, func):
        """Return flat list of rendered text lines from all HIR nodes."""
        lines = []
        for node in func.hir:
            lines.extend(text for _, text in node.render())
        return lines

    def test_basic_overwrite_kills_param_name(self):
        """
        Proto: flags1→R7, osd_addr→R2R3
        HIR:  A = R7; R7 = A; XRAM[X] = R7
        After AccumRelay collapses A = R7; R7 = A to R7 = R7 (no-op), then
        the explicit Assign(R7, Reg("R2")) overwrite means XRAM[X] = R7 should
        NOT be substituted with flags1.
        """
        PROTOTYPES["f_kill1"] = FuncProto(
            return_type="void",
            params=[
                Param("flags1", "uint8_t", ("R7",)),
                Param("osd_addr", "uint16_t", ("R2", "R3")),
            ],
        )
        # Simulate: R7 gets overwritten with R2 (osd_addr.hi), then used
        nodes = [
            Assign(0x1000, Reg("R7"), Reg("R2")),          # R7 = osd_addr.hi
            Assign(0x1002, XRAMRef(Name("OSD_ADDR_MSB")), Reg("R7")),  # use R7
        ]
        func = make_single_block_func("f_kill1", nodes)
        TypeAwareSimplifier().run(func)
        lines = self._rendered(func)
        # R7 store should use osd_addr name, not flags1
        assert not any("flags1" in l for l in lines), \
            f"flags1 leaked after overwrite: {lines}"

    def test_two_sequential_overwrites(self):
        """
        Proto: flags1→R7, osd_addr→R2R3
        HIR:
          R7 = R2                          (overwrite 1: R7 = osd_addr.hi)
          XRAM[OSD_ADDR_MSB] = R7          (use 1)
          R7 = R3                          (overwrite 2: R7 = osd_addr.lo)
          XRAM[OSD_ADDR_LSB] = R7          (use 2)
        Both XRAM stores must NOT contain flags1.
        """
        PROTOTYPES["f_kill2"] = FuncProto(
            return_type="void",
            params=[
                Param("flags1", "uint8_t", ("R7",)),
                Param("osd_addr", "uint16_t", ("R2", "R3")),
            ],
        )
        nodes = [
            Assign(0x1000, Reg("R7"), Reg("R2")),
            Assign(0x1002, XRAMRef(Name("OSD_ADDR_MSB")), Reg("R7")),
            Assign(0x1004, Reg("R7"), Reg("R3")),
            Assign(0x1006, XRAMRef(Name("OSD_ADDR_LSB")), Reg("R7")),
        ]
        func = make_single_block_func("f_kill2", nodes)
        TypeAwareSimplifier().run(func)
        lines = self._rendered(func)
        assert not any("flags1" in l for l in lines), \
            f"flags1 leaked after overwrite: {lines}"
