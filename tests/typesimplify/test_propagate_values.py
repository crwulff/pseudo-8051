"""
tests/typesimplify/test_propagate_values.py — Tests for _propagate_values pass.
"""

import pytest

from pseudo8051.passes.typesimplify._post import _propagate_values
from pseudo8051.passes.patterns._utils import VarInfo
from pseudo8051.ir.hir import Assign, ExprStmt, Statement
from pseudo8051.ir.expr import Reg, Name, XRAMRef


class TestPropagateValues:

    def test_dptr_fold_into_statement(self):
        """DPTR=Name("offset") folded into call Statement."""
        nodes = [
            Assign(0, Reg("DPTR"), Name("offset")),
            Statement(1, "retval1 = func(DPTR);"),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        assert "DPTR" not in result[0].text
        assert "offset" in result[0].text

    def test_dptr_fold_into_xram_lhs(self):
        """DPTR=Name("_dest") folded into XRAM[DPTR]=val LHS."""
        nodes = [
            Assign(0, Reg("DPTR"), Name("_dest")),
            Assign(1, XRAMRef(Reg("DPTR")), Name("val")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        assert result[0].lhs.render() == "XRAM[_dest]"

    def test_r7_fold_into_xram_rhs(self):
        """R7=Name("retval1") folded into XRAM[DPTR]=R7 rhs."""
        nodes = [
            Assign(0, Reg("R7"), Name("retval1")),
            Assign(1, XRAMRef(Reg("DPTR")), Reg("R7")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        assert result[0].rhs.render() == "retval1"

    def test_multiple_uses_not_folded(self):
        """Register with >1 downstream uses is not folded."""
        nodes = [
            Assign(0, Reg("R7"), Name("val")),
            Assign(1, Reg("R6"), Reg("R7")),          # use 1
            Assign(2, XRAMRef(Reg("DPTR")), Reg("R7")),  # use 2
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 3  # nothing removed

    def test_retval_single_use_inlined(self):
        """Single-use retval Statement is inlined into the Assign target."""
        nodes = [
            Statement(0, "int8_t retval1 = func(x);"),
            Assign(1, XRAMRef(Name("dest")), Name("retval1")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        assert isinstance(result[0], Statement)
        assert "func(x)" in result[0].text
        assert "retval1" not in result[0].text

    def test_retval_multiple_uses_not_inlined(self):
        """retval used twice → not inlined."""
        nodes = [
            Statement(0, "int8_t retval1 = func(x);"),
            Assign(1, Reg("R7"), Name("retval1")),
            Assign(2, Reg("R6"), Name("retval1")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 3

    def test_full_chain(self):
        """DPTR=off; Statement(retval1=call(...,DPTR)); R7=retval1; DPTR=_dest;
           XRAM[DPTR]=R7 → XRAM[_dest]=call(...,off)."""
        nodes = [
            Assign(0, Reg("DPTR"), Name("offset")),
            Statement(1, "int8_t retval1 = code_7_read(a, b, DPTR);"),
            Assign(2, Reg("R7"), Name("retval1")),
            Assign(3, Reg("DPTR"), Name("_dest")),
            Assign(4, XRAMRef(Reg("DPTR")), Reg("R7")),
        ]
        result = _propagate_values(nodes, {})
        assert len(result) == 1
        assert isinstance(result[0], Statement)
        t = result[0].text
        assert "offset" in t
        assert "DPTR" not in t
        assert "retval1" not in t
        assert "R7" not in t
        assert "XRAM[_dest]" in t
        assert "code_7_read" in t
