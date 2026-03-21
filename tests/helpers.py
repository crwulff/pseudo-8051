"""
tests/helpers.py — Lightweight BasicBlock / Function substitutes for testing.

No IDA API is called; these classes expose only the attributes and properties
that the structural passes and TypeAwareSimplifier actually access.
"""

from __future__ import annotations
from typing import List, Optional


class FakeBlock:
    """BasicBlock substitute — no IDA required."""

    def __init__(self, ea: int, hir=None, label: Optional[str] = None,
                 live_in=frozenset()):
        self.start_ea       = ea
        self.end_ea         = ea + 0x10
        self.hir            = list(hir or [])
        self.label          = label
        self.live_in        = live_in
        self.live_out       = frozenset()
        self.cp_entry       = None
        self.is_loop_header = False
        self._absorbed      = False
        self._preds: List[FakeBlock] = []
        self._succs: List[FakeBlock] = []
        self._func          = None   # set by FakeFunction.__init__

    @property
    def predecessors(self) -> List[FakeBlock]:
        return self._preds

    @property
    def successors(self) -> List[FakeBlock]:
        return self._succs


class FakeFunction:
    """Function substitute — no IDA required."""

    def __init__(self, name: str, blocks: List[FakeBlock],
                 hir=None):
        self.name       = name
        self.ea         = blocks[0].start_ea if blocks else 0
        self._blocks    = list(blocks)
        self._block_map = {b.start_ea: b for b in blocks}
        self.hir        = list(hir or [])
        for b in blocks:
            b._func = self

    @property
    def blocks(self) -> List[FakeBlock]:
        return self._blocks

    @property
    def entry_block(self) -> FakeBlock:
        return self._block_map[self.ea]


def connect(src: FakeBlock, dst: FakeBlock) -> None:
    """Add a directed CFG edge src → dst."""
    src._succs.append(dst)
    dst._preds.append(src)


def make_single_block_func(name: str, stmts_or_texts,
                           proto=None):
    """
    Build a single-block FakeFunction from a list of HIRNodes or raw strings.

    Strings are wrapped in Statement with an incrementing ea starting at 0x1000.
    HIRNodes are used as-is (ea attribute preserved).
    If *proto* is provided it is inserted into PROTOTYPES[name].
    """
    from pseudo8051.ir.hir import Statement, HIRNode
    from pseudo8051.prototypes import PROTOTYPES

    stmts = []
    for i, item in enumerate(stmts_or_texts):
        if isinstance(item, HIRNode):
            stmts.append(item)
        else:
            stmts.append(Statement(0x1000 + i * 2, item))
    block = FakeBlock(0x1000, hir=stmts)
    func  = FakeFunction(name, [block], hir=stmts)
    if proto is not None:
        PROTOTYPES[name] = proto
    return func
