"""
tests/test_patterns.py — Unit tests for individual Pattern classes.

All tests operate purely on Statement nodes and VarInfo dicts; no IDA API
calls are made.  The simplify callback is a no-op lambda.
"""

import pytest
from pseudo8051.ir.hir import Statement
from pseudo8051.passes.patterns._utils import VarInfo, _replace_single_regs, _replace_pairs
from pseudo8051.passes.patterns.accum_relay   import AccumRelayPattern
from pseudo8051.passes.patterns.const_group   import ConstGroupPattern
from pseudo8051.passes.patterns.neg16         import Neg16Pattern
from pseudo8051.passes.patterns.reg_copy_group import RegCopyGroupPattern

# Dummy simplify callback that returns nodes unchanged
_noop_simplify = lambda nodes, reg_map: nodes


# ── AccumRelayPattern ─────────────────────────────────────────────────────────

class TestAccumRelayPattern:
    def _pat(self):
        return AccumRelayPattern()

    def test_basic_no_subst(self):
        """A = R7; XRAM[sym] = A; → XRAM[sym] = R7; (no reg_map)"""
        nodes = [Statement(0, "A = R7;"), Statement(2, "XRAM[sym] = A;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 2
        assert len(replacement) == 1
        assert replacement[0].text == "XRAM[sym] = R7;"

    def test_with_param_substitution(self):
        """A = R7; XRAM[sym] = A; with R7=H param → XRAM[sym] = H;"""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",), is_param=True)}
        nodes = [Statement(0, "A = R7;"), Statement(2, "XRAM[sym] = A;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        assert result[0][0].text == "XRAM[sym] = H;"

    def test_no_fire_second_not_a_read(self):
        """Second statement doesn't read A → pattern must not fire."""
        nodes = [Statement(0, "A = R7;"), Statement(2, "XRAM[sym] = R6;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_trivial_a_eq_a(self):
        """A = A; ... → skip (expr == 'A')."""
        nodes = [Statement(0, "A = A;"), Statement(2, "XRAM[sym] = A;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_target_is_a(self):
        """A = R7; A = A; → skip (target is 'A')."""
        nodes = [Statement(0, "A = R7;"), Statement(2, "A = A;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_no_fire_only_one_node(self):
        """Only one node → no pair to collapse."""
        nodes = [Statement(0, "A = R7;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is None

    def test_ea_preserved(self):
        """Output Statement carries the ea of the first node."""
        nodes = [Statement(0x1234, "A = R6;"), Statement(0x1236, "R7 = A;")]
        result = self._pat().match(nodes, 0, {}, _noop_simplify)
        assert result is not None
        assert result[0][0].ea == 0x1234

    def test_with_pair_substitution(self):
        """Target uses a reg pair that gets substituted."""
        vinfo = VarInfo("myvar", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        # A = R5; R6R7 = A; → R6R7 = R5; → myvar = R5;
        nodes = [Statement(0, "A = R5;"), Statement(2, "R6R7 = A;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        # _replace_pairs replaces R6R7 on the LHS too (it's not an assignment
        # in the sense that A = R5 is the payload; the new text is
        # "R6R7 = R5;" which gets _replace_pairs applied → "myvar = R5;"
        assert result[0][0].text == "myvar = R5;"


# ── ConstGroupPattern ─────────────────────────────────────────────────────────

class TestConstGroupPattern:
    def _pat(self):
        return ConstGroupPattern()

    def _u32_reg_map(self, name="dividend"):
        vinfo = VarInfo(name, "uint32_t", ("R4", "R5", "R6", "R7"))
        return {
            "R4R5R6R7": vinfo, "R4": vinfo, "R5": vinfo,
            "R6": vinfo, "R7": vinfo,
        }

    def test_u32_const_group(self):
        """5 stmts loading 0x00005dc0 into R4R5R6R7."""
        reg_map = self._u32_reg_map()
        nodes = [
            Statement(0, "A = 0x00;"),
            Statement(2, "R4 = A;"),
            Statement(4, "R5 = A;"),
            Statement(6, "R6 = 0x5d;"),
            Statement(8, "R7 = 0xc0;"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 5
        assert len(replacement) == 1
        assert replacement[0].text == "uint32_t dividend = 0x00005dc0;"

    def test_u32_fold_into_call(self):
        """Const group immediately followed by return with the pair → fold."""
        reg_map = self._u32_reg_map()
        nodes = [
            Statement(0, "A = 0x00;"),
            Statement(2, "R4 = A;"),
            Statement(4, "R5 = A;"),
            Statement(6, "R6 = 0x5d;"),
            Statement(8, "R7 = 0xc0;"),
            Statement(10, "return func(R4R5R6R7);"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 6   # consumed 5 group stmts + 1 return
        assert replacement[0].text == "return func(0x00005dc0);"

    def test_u16_const_group(self):
        """2-byte constant into R6R7."""
        vinfo = VarInfo("val", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        nodes = [
            Statement(0, "R6 = 0x12;"),
            Statement(2, "R7 = 0x34;"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        assert result[0][0].text == "uint16_t val = 0x1234;"

    def test_no_fire_incomplete_group(self):
        """Only R6 loaded, R7 missing → no match."""
        vinfo = VarInfo("val", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        nodes = [Statement(0, "R6 = 0x12;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_single_reg_var(self):
        """Single-byte VarInfo (regs < 2) → pattern skips it."""
        vinfo = VarInfo("x", "uint8_t", ("R7",))
        reg_map = {"R7": vinfo}
        nodes = [Statement(0, "R7 = 0x42;")]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None


# ── Neg16Pattern ──────────────────────────────────────────────────────────────

class TestNeg16Pattern:
    def _pat(self):
        return Neg16Pattern()

    def _neg_nodes(self, r_lo="R7", r_hi="R6"):
        return [
            Statement(0,  "C = 0;"),
            Statement(2,  "A = 0;"),
            Statement(4,  f"A -= {r_lo} + C;"),
            Statement(6,  f"{r_lo} = A;"),
            Statement(8,  "A = 0;"),
            Statement(10, f"A -= {r_hi} + C;"),
            Statement(12, f"{r_hi} = A;"),
        ]

    def test_basic_neg16(self):
        """7-statement SUBB negation collapses to 'x = -x;'."""
        vinfo = VarInfo("x", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = self._neg_nodes()
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is not None
        replacement, new_i = result
        assert new_i == 7
        assert replacement[0].text == "x = -x;"

    def test_no_fire_wrong_var_order(self):
        """lo and hi from different VarInfo → no match."""
        vlo = VarInfo("a", "uint8_t", ("R7",))
        vhi = VarInfo("b", "uint8_t", ("R6",))
        reg_map = {"R7": vlo, "R6": vhi}
        nodes = self._neg_nodes()
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_too_few_nodes(self):
        """Fewer than 7 nodes → no match."""
        vinfo = VarInfo("x", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = self._neg_nodes()[:6]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None

    def test_no_fire_mismatched_store(self):
        """Store register doesn't match SUBB source → no match."""
        vinfo = VarInfo("x", "int16_t", ("R6", "R7"))
        reg_map = {"R6": vinfo, "R7": vinfo}
        nodes = [
            Statement(0,  "C = 0;"),
            Statement(2,  "A = 0;"),
            Statement(4,  "A -= R7 + C;"),
            Statement(6,  "R6 = A;"),   # should be R7
            Statement(8,  "A = 0;"),
            Statement(10, "A -= R6 + C;"),
            Statement(12, "R6 = A;"),
        ]
        result = self._pat().match(nodes, 0, reg_map, _noop_simplify)
        assert result is None


# ── _replace_single_regs ─────────────────────────────────────────────────────

class TestReplaceSingleRegs:
    def test_rhs_only_substitution(self):
        """LHS stays raw; RHS gets param name substituted."""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",), is_param=True)}
        result = _replace_single_regs("R7 = R7;", reg_map)
        assert result == "R7 = H;"

    def test_non_param_not_substituted(self):
        """Entries without is_param=True are left alone."""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",))}  # is_param defaults False
        result = _replace_single_regs("R7 = R7;", reg_map)
        assert result == "R7 = R7;"

    def test_no_assignment_substitutes_everywhere(self):
        """In a non-assignment expression, all occurrences are substituted."""
        reg_map = {"R7": VarInfo("H", "uint8_t", ("R7",), is_param=True)}
        result = _replace_single_regs("return R7;", reg_map)
        assert result == "return H;"

    def test_xram_entry_skipped(self):
        """XRAM-local VarInfo (xram_sym set) must not be substituted."""
        vinfo = VarInfo("var1", "uint16_t", (), xram_sym="EXT_DC8A")
        reg_map = {"R7": vinfo}   # degenerate: xram entry keyed by reg name
        # is_param=False, so it won't match anyway, but xram_sym check also guards
        result = _replace_single_regs("foo = R7;", reg_map)
        assert result == "foo = R7;"


# ── _replace_pairs ────────────────────────────────────────────────────────────

class TestReplacePairs:
    def test_basic_pair_substitution(self):
        """R6R7 replaced by variable name in expression."""
        vinfo = VarInfo("myvar", "uint16_t", ("R6", "R7"))
        reg_map = {"R6R7": vinfo, "R6": vinfo, "R7": vinfo}
        result = _replace_pairs("foo(R6R7)", reg_map)
        assert result == "foo(myvar)"

    def test_xram_pair_skipped(self):
        """XRAM-local pair (xram_sym set) is not substituted."""
        vinfo = VarInfo("local1", "uint16_t", ("R6", "R7"),
                        xram_sym="EXT_DC00")
        reg_map = {"R6R7": vinfo}
        result = _replace_pairs("foo(R6R7)", reg_map)
        assert result == "foo(R6R7)"

    def test_longer_key_wins(self):
        """4-byte pair replaced before 2-byte pair (longest-first)."""
        v4 = VarInfo("quad", "uint32_t", ("R4", "R5", "R6", "R7"))
        v2 = VarInfo("pair", "uint16_t", ("R6", "R7"))
        reg_map = {
            "R4R5R6R7": v4, "R4": v4, "R5": v4, "R6R7": v2,
        }
        result = _replace_pairs("f(R4R5R6R7)", reg_map)
        assert result == "f(quad)"

    def test_single_reg_key_ignored(self):
        """Keys of length ≤ 2 (single registers like 'R7') are skipped."""
        vinfo = VarInfo("H", "uint8_t", ("R7",), is_param=True)
        reg_map = {"R7": vinfo}
        result = _replace_pairs("foo(R7)", reg_map)
        assert result == "foo(R7)"


# ── RegCopyGroupPattern ───────────────────────────────────────────────────────

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
