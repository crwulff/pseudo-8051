"""
tests/passes/test_fixup_jmptable_edges.py — Tests for the jump-table CFG fixup.

Tests the _add_successor override mechanism on FakeBlock and the
fixup_jmptable_edges() function (with IDA helpers patched out).
"""

import pytest
from unittest.mock import patch

from tests.helpers import FakeBlock, FakeFunction, connect
from pseudo8051.ir.hir  import Assign, Label
from pseudo8051.ir.expr import Reg, Const
from pseudo8051.passes.jmptable import fixup_jmptable_edges


# ── _add_successor / override mechanism ──────────────────────────────────────

class TestAddSuccessor:

    def test_bidirectional(self):
        a = FakeBlock(0x100)
        b = FakeBlock(0x200)
        a._add_successor(b)
        assert b in a.successors
        assert a in b.predecessors

    def test_does_not_affect_base_lists(self):
        a = FakeBlock(0x100)
        b = FakeBlock(0x200)
        a._add_successor(b)
        assert a._succs == []
        assert b._preds == []
        assert a._succ_extra == [b]
        assert b._pred_extra == [a]

    def test_idempotent(self):
        a = FakeBlock(0x100)
        b = FakeBlock(0x200)
        a._add_successor(b)
        a._add_successor(b)
        assert a.successors.count(b) == 1
        assert b.predecessors.count(a) == 1

    def test_does_not_clobber_existing_succs(self):
        a = FakeBlock(0x100)
        b = FakeBlock(0x200)
        c = FakeBlock(0x300)
        connect(a, b)           # normal CFG edge
        a._add_successor(c)     # synthetic edge
        assert b in a.successors
        assert c in a.successors

    def test_multiple_synthetic_succs(self):
        jmp = FakeBlock(0x100)
        s0  = FakeBlock(0x200)
        s1  = FakeBlock(0x202)
        s2  = FakeBlock(0x204)
        jmp._add_successor(s0)
        jmp._add_successor(s1)
        jmp._add_successor(s2)
        assert jmp.successors == [s0, s1, s2]
        for s in (s0, s1, s2):
            assert jmp in s.predecessors


# ── fixup_jmptable_edges ──────────────────────────────────────────────────────

def _make_computed_jump_hir(jmp_ea: int):
    """A minimal ComputedJump node (Assign DPTR + ComputedJump stand-in)."""
    from pseudo8051.ir.hir import ComputedJump
    return [
        Assign(jmp_ea - 3, Reg("DPTR"), Const(0x1326)),
        ComputedJump(jmp_ea),
    ]


class TestFixupJmptableEdges:

    def _make_func(self, jmp_ea, sjmp_eas, jmp_has_succs=False):
        """Build a FakeFunction with a JMP block and SJMP dispatch blocks."""
        jmp_block = FakeBlock(jmp_ea, hir=_make_computed_jump_hir(jmp_ea))
        sjmp_blocks = [FakeBlock(ea) for ea in sjmp_eas]

        if jmp_has_succs:
            for s in sjmp_blocks:
                connect(jmp_block, s)

        all_blocks = [jmp_block] + sjmp_blocks
        func = FakeFunction("test_func", all_blocks)
        return func, jmp_block, sjmp_blocks

    def test_does_nothing_when_succs_present(self):
        """If block already has successors, no extra edges should be added."""
        # Single-block function: the JMP block has two existing CFG successors.
        case_a = FakeBlock(0x1326)
        case_b = FakeBlock(0x1328)
        jmp_block = FakeBlock(0x713c3, hir=_make_computed_jump_hir(0x713c3))
        connect(jmp_block, case_a)
        connect(jmp_block, case_b)
        func = FakeFunction("test_func", [jmp_block, case_a, case_b])

        with patch("pseudo8051.passes.jmptable._find_jmp_hir_idx",
                   return_value=None) as mock_find:
            fixup_jmptable_edges(func)
            # jmp_block has succs → skipped; case_a/case_b have no ComputedJump
            # so _find_jmp_hir_idx returns None for them
        assert jmp_block._succ_extra == []

    def test_wires_sjmp_blocks(self):
        """Blocks with ComputedJump + empty succs get synthetic edges added."""
        stride = 2
        table_ea = 0x1326
        sjmp_eas = [table_ea + i * stride for i in range(3)]
        func, jmp_block, sjmp_blocks = self._make_func(0x713c3, sjmp_eas)

        cases = [([i], f"label_{i}") for i in range(3)]

        with patch("pseudo8051.passes.jmptable._find_jmp_hir_idx", return_value=1), \
             patch("pseudo8051.passes.jmptable._get_table_ea", return_value=table_ea), \
             patch("pseudo8051.passes.jmptable._read_jump_table",
                   return_value=(cases, stride, sjmp_eas)):
            fixup_jmptable_edges(func)

        assert set(jmp_block.successors) == set(sjmp_blocks)
        for s in sjmp_blocks:
            assert jmp_block in s.predecessors

    def test_skips_block_with_no_computed_jump(self):
        """Block with no ComputedJump (find returns None) gets no edges."""
        func, jmp_block, sjmp_blocks = self._make_func(0x713c3, [0x1326, 0x1328])

        with patch("pseudo8051.passes.jmptable._find_jmp_hir_idx", return_value=None):
            fixup_jmptable_edges(func)

        assert jmp_block.successors == []

    def test_skips_when_table_ea_unknown(self):
        """Block where DPTR value is unknown gets no edges."""
        func, jmp_block, sjmp_blocks = self._make_func(0x713c3, [0x1326, 0x1328])

        with patch("pseudo8051.passes.jmptable._find_jmp_hir_idx", return_value=1), \
             patch("pseudo8051.passes.jmptable._get_table_ea", return_value=None):
            fixup_jmptable_edges(func)

        assert jmp_block.successors == []

    def test_skips_when_read_table_empty(self):
        """Block where table read returns empty gets no edges."""
        func, jmp_block, sjmp_blocks = self._make_func(0x713c3, [0x1326, 0x1328])

        with patch("pseudo8051.passes.jmptable._find_jmp_hir_idx", return_value=1), \
             patch("pseudo8051.passes.jmptable._get_table_ea", return_value=0x1326), \
             patch("pseudo8051.passes.jmptable._read_jump_table", return_value=([], 0, [])):
            fixup_jmptable_edges(func)

        assert jmp_block.successors == []

    def test_ignores_missing_sjmp_block(self):
        """If a table entry's block isn't in _block_map, silently skip it."""
        table_ea = 0x1326
        stride = 2
        # Only one of the three SJMP EAs is in the func's block_map
        sjmp_eas = [table_ea]   # only 0x1326 present, 0x1328/0x132a missing
        func, jmp_block, sjmp_blocks = self._make_func(0x713c3, sjmp_eas)

        cases = [([i], f"label_{i}") for i in range(3)]
        entry_eas = [table_ea + i * stride for i in range(3)]
        with patch("pseudo8051.passes.jmptable._find_jmp_hir_idx", return_value=1), \
             patch("pseudo8051.passes.jmptable._get_table_ea", return_value=table_ea), \
             patch("pseudo8051.passes.jmptable._read_jump_table",
                   return_value=(cases, stride, entry_eas)):
            fixup_jmptable_edges(func)

        # Only the one present block gets wired
        assert jmp_block.successors == sjmp_blocks
