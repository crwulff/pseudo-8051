"""
Tests for DeadAssignEliminator: backward intra-block dead assignment removal.
"""

import pytest

from pseudo8051.ir.hir         import Assign, HIRNode
from pseudo8051.ir.hir.expr_stmt       import ExprStmt
from pseudo8051.ir.hir.compound_assign import CompoundAssign
from pseudo8051.ir.hir.if_goto        import IfGoto
from pseudo8051.ir.expr        import Reg, Const, BinOp, XRAMRef, IRAMRef, Call
from pseudo8051.passes.dead_assign import DeadAssignEliminator

from tests.helpers import FakeBlock, FakeFunction


def _func(nodes, live_out=frozenset()):
    block = FakeBlock(0x1000, hir=list(nodes))
    block.live_out = live_out
    return FakeFunction("f", [block]), block


# ── Basic removal ─────────────────────────────────────────────────────────────

def test_dead_assign_removed():
    """A = R0 where A is not live → removed."""
    func, block = _func([Assign(0x1000, Reg("A"), Reg("R0"))], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert block.hir == []


def test_live_assign_kept():
    """A = R0 where A is live at live_out → kept."""
    func, block = _func([Assign(0x1000, Reg("A"), Reg("R0"))], live_out=frozenset({"A"}))
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1


def test_used_within_block_is_not_removed():
    """A = R0; XRAM[0x100] = A → A is used, so assignment kept."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, XRAMRef(Const(0x100)), Reg("A")),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 2


# ── Cascade removal ───────────────────────────────────────────────────────────

def test_cascade_removal():
    """A = R0; B = A; both dead → both removed in single backward pass."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, Reg("B"), Reg("A")),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert block.hir == []


def test_cascade_partial():
    """A = R0; B = A; C = A; B dead but A needed by C → B removed, A and C kept."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, Reg("B"), Reg("A")),
        Assign(0x1004, Reg("C"), Reg("A")),
    ], live_out=frozenset({"C"}))
    DeadAssignEliminator().run(func)
    # B is removed (dead); A is kept (used by C); C is kept (in live_out)
    assert len(block.hir) == 2
    texts = [n.render()[0][1].strip() for n in block.hir]
    assert "B = A;" not in texts
    assert "A = R0;" in texts
    assert "C = A;" in texts


# ── Non-removable cases ───────────────────────────────────────────────────────

def test_memory_write_never_removed():
    """XRAM[addr] = R0 is never removed — LHS is not a Reg."""
    func, block = _func([
        Assign(0x1000, XRAMRef(Const(0x100)), Reg("R0")),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1


def test_memory_read_rhs_not_removed():
    """A = XRAM[addr] where A is dead — kept because XRAMRef is not pure."""
    func, block = _func([
        Assign(0x1000, Reg("A"), XRAMRef(Const(0x100))),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1


def test_iram_read_rhs_not_removed():
    """A = IRAM[addr] where A is dead — kept because IRAMRef is not pure."""
    from pseudo8051.ir.expr import IRAMRef
    func, block = _func([
        Assign(0x1000, Reg("A"), IRAMRef(Const(0x20))),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1


def test_call_result_not_removed():
    """A = call() where A is dead — kept because Call is not pure."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Call("foo", [])),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1


def test_expr_stmt_never_removed():
    """ExprStmt(call()) is never removed — it's not an Assign."""
    func, block = _func([
        ExprStmt(0x1000, Call("foo", [])),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1


def test_compound_assign_never_removed():
    """A &= 0x0F where A is dead — kept because it's CompoundAssign, not Assign."""
    func, block = _func([
        CompoundAssign(0x1000, Reg("A"), "&=", Const(0x0F)),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1


def test_reggroup_lhs_not_removed():
    """Assign(RegGroup(("R6","R7")), ...) is not removed even if dead (non-single Regs)."""
    from pseudo8051.ir.expr import RegGroup
    func, block = _func([
        Assign(0x1000, RegGroup(("R6", "R7")), Const(0)),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    # RegGroup is not a single Reg, so is_single is False → not removed
    assert len(block.hir) == 1


# ── IfGoto interaction ────────────────────────────────────────────────────────

def test_ifgoto_keeps_used_assign():
    """A = R0; if (A == 0) → A is used in condition, assignment kept."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        IfGoto(0x1002, BinOp(Reg("A"), "==", Const(0)), "L_1002"),
    ], live_out=frozenset())
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 2


# ── live_out boundary ─────────────────────────────────────────────────────────

def test_live_at_exit_prevents_removal():
    """A = R0; A is live at live_out (needed by successor) → kept."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
    ], live_out=frozenset({"A"}))
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1


def test_dead_at_exit_after_rewrite():
    """A = R0; A = R1; first assignment is dead after rewrite → removed."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),   # dead: A overwritten immediately
        Assign(0x1002, Reg("A"), Reg("R1")),   # live at exit
    ], live_out=frozenset({"A"}))
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1
    assert block.hir[0].rhs == Reg("R1")


# ── Absorbed blocks ───────────────────────────────────────────────────────────

def test_absorbed_block_skipped():
    """Absorbed blocks are not touched."""
    block = FakeBlock(0x1000, hir=[Assign(0x1000, Reg("A"), Reg("R0"))])
    block.live_out = frozenset()
    block._absorbed = True
    func = FakeFunction("f", [block])
    DeadAssignEliminator().run(func)
    assert len(block.hir) == 1   # untouched
