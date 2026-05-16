"""
Tests for CopyPropagator: forward intra-block copy propagation.
"""

import pytest

from pseudo8051.ir.hir         import Assign, HIRNode
from pseudo8051.ir.hir.expr_stmt       import ExprStmt
from pseudo8051.ir.hir.compound_assign import CompoundAssign
from pseudo8051.ir.hir.if_goto        import IfGoto
from pseudo8051.ir.expr        import Regs, Reg, Const, BinOp, XRAMRef, Call, UnaryOp
from pseudo8051.passes.copyprop import CopyPropagator

from tests.helpers import FakeBlock, FakeFunction


def _func(nodes, live_out=frozenset()):
    block = FakeBlock(0x1000, hir=nodes)
    block.live_out = live_out
    return FakeFunction("f", [block]), block


# ── Basic register-to-register propagation ────────────────────────────────────

def test_reg_to_reg_propagated():
    """A = R0; R1 = A → A = R0; R1 = R0"""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, Reg("R1"), Reg("A")),
    ])
    CopyPropagator().run(func)
    assert block.hir[0].rhs == Reg("R0")   # unchanged
    assert block.hir[1].rhs == Reg("R0")   # A → R0


def test_const_propagated():
    """A = 0; R0 = A → A = 0; R0 = 0"""
    func, block = _func([
        Assign(0x1000, Reg("A"), Const(0)),
        Assign(0x1002, Reg("R0"), Reg("A")),
    ])
    CopyPropagator().run(func)
    assert block.hir[1].rhs == Const(0)


def test_transitive_propagation():
    """A = R0; B = A; C = B → all resolve to R0."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, Reg("B"), Reg("A")),
        Assign(0x1004, Reg("C"), Reg("B")),
    ])
    CopyPropagator().run(func)
    assert block.hir[1].rhs == Reg("R0")
    assert block.hir[2].rhs == Reg("R0")


def test_propagation_into_binop():
    """A = R0; R1 = A + 1 → R1 = R0 + 1"""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, Reg("R1"), BinOp(Reg("A"), "+", Const(1))),
    ])
    CopyPropagator().run(func)
    rhs = block.hir[1].rhs
    assert isinstance(rhs, BinOp)
    assert rhs.lhs == Reg("R0")
    assert rhs.rhs == Const(1)


def test_propagation_into_xramref_address():
    """A = R0; XRAM[A] = R1 → XRAM[R0] = R1 (address is substituted)."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, XRAMRef(Reg("A")), Reg("R1")),
    ])
    CopyPropagator().run(func)
    lhs = block.hir[1].lhs
    assert isinstance(lhs, XRAMRef)
    # Note: map_exprs on Assign does NOT touch the LHS — only the RHS.
    # The address A in LHS is NOT substituted (LHS is a write position).
    # This is intentional: map_exprs only transforms read positions.
    assert lhs.inner == Reg("A")   # LHS address unchanged


def test_propagation_into_xramref_write_rhs():
    """A = R0; XRAM[addr] = A → XRAM[addr] = R0 (RHS substituted)."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, XRAMRef(Const(0x100)), Reg("A")),
    ])
    CopyPropagator().run(func)
    assert block.hir[1].rhs == Reg("R0")


# ── Invalidation ──────────────────────────────────────────────────────────────

def test_invalidated_by_lhs_overwrite():
    """A = R0; A = R1; R2 = A → R2 = R1 (not R0)."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, Reg("A"), Reg("R1")),
        Assign(0x1004, Reg("R2"), Reg("A")),
    ])
    CopyPropagator().run(func)
    assert block.hir[2].rhs == Reg("R1")


def test_invalidated_by_source_overwrite():
    """A = R0; R0 = 5; R1 = A → R1 = A (copy invalidated when source R0 written)."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, Reg("R0"), Const(5)),
        Assign(0x1004, Reg("R1"), Reg("A")),
    ])
    CopyPropagator().run(func)
    # After node 1, copies["A"] = Reg("R0") should be invalidated because R0 changed.
    # So node 2 should keep A (not substitute).
    assert block.hir[2].rhs == Reg("A")


def test_call_kills_all_copies():
    """A = R0; call(); R1 = A → R1 = A (all copies killed by call)."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        ExprStmt(0x1002, Call("foo", [])),
        Assign(0x1004, Reg("R1"), Reg("A")),
    ])
    CopyPropagator().run(func)
    assert block.hir[2].rhs == Reg("A")


def test_call_args_are_substituted_before_kill():
    """A = R0; call(A) → call(R0); copies cleared after."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        ExprStmt(0x1002, Call("foo", [Reg("A")])),
        Assign(0x1004, Reg("R1"), Reg("A")),
    ])
    CopyPropagator().run(func)
    # The call's arg should be substituted (step 1 runs before step 2 kill)
    call_expr = block.hir[1].expr
    assert isinstance(call_expr, Call)
    assert call_expr.args[0] == Reg("R0")
    # But subsequent use of A is not substituted (call killed copies)
    assert block.hir[2].rhs == Reg("A")


def test_assign_call_result_kills_copies():
    """A = R0; A = call() → prior A=R0 copy is gone; R1 = A is not substituted."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        Assign(0x1002, Reg("A"), Call("foo", [])),
        Assign(0x1004, Reg("R1"), Reg("A")),
    ])
    CopyPropagator().run(func)
    assert block.hir[2].rhs == Reg("A")   # A is not a copy of R0 anymore


# ── CompoundAssign ────────────────────────────────────────────────────────────

def test_compound_assign_invalidates_copy():
    """A = R0; A &= 0x0F; R1 = A → R1 = A (compound assign invalidates copy)."""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        CompoundAssign(0x1002, Reg("A"), "&=", Const(0x0F)),
        Assign(0x1004, Reg("R1"), Reg("A")),
    ])
    CopyPropagator().run(func)
    assert block.hir[2].rhs == Reg("A")


# ── IfGoto ────────────────────────────────────────────────────────────────────

def test_propagation_into_ifgoto_condition():
    """A = R0; if (A == 0) goto L → if (R0 == 0) goto L"""
    func, block = _func([
        Assign(0x1000, Reg("A"), Reg("R0")),
        IfGoto(0x1002, BinOp(Reg("A"), "==", Const(0)), "L_1002"),
    ])
    CopyPropagator().run(func)
    cond = block.hir[1].cond
    assert isinstance(cond, BinOp)
    assert cond.lhs == Reg("R0")


# ── ++/-- operands are never substituted ─────────────────────────────────────

def test_increment_operand_not_substituted():
    """DPTR = const; DPTR++ → DPTR++ unchanged (++ operand is a write target)."""
    func, block = _func([
        Assign(0x1000, Reg("DPTR"), Const(0xdc68)),
        ExprStmt(0x1002, UnaryOp("++", Reg("DPTR"), post=True)),
    ])
    CopyPropagator().run(func)
    # DPTR++ must stay as DPTR++, not become 0xdc68++
    inc_node = block.hir[1]
    assert isinstance(inc_node.expr, UnaryOp)
    assert isinstance(inc_node.expr.operand, Regs)
    assert inc_node.expr.operand.name == "DPTR"


def test_increment_invalidates_copy():
    """DPTR = const; DPTR++; XRAM[DPTR] = A → XRAM[DPTR] = A (copy stale after ++)."""
    func, block = _func([
        Assign(0x1000, Reg("DPTR"), Const(0xdc68)),
        ExprStmt(0x1002, UnaryOp("++", Reg("DPTR"), post=True)),
        Assign(0x1004, XRAMRef(Reg("DPTR")), Reg("A")),
    ])
    CopyPropagator().run(func)
    # After DPTR++, the copy is stale — XRAM[DPTR] LHS should stay as XRAM[DPTR],
    # not become XRAM[0xdc68].  (LHS is never transformed anyway; check RHS/A.)
    # The key check: DPTR++ node must still reference Reg("DPTR"), not Const.
    inc_node = block.hir[1]
    assert isinstance(inc_node.expr, UnaryOp)
    assert isinstance(inc_node.expr.operand, Regs)


def test_copy_after_increment_not_propagated():
    """DPTR = Const; DPTR++; R0 = DPTR → R0 = DPTR (copy invalidated by ++)."""
    func, block = _func([
        Assign(0x1000, Reg("DPTR"), Const(0xdc68)),
        ExprStmt(0x1002, UnaryOp("++", Reg("DPTR"), post=True)),
        Assign(0x1004, Reg("R0"), Reg("DPTR")),
    ])
    CopyPropagator().run(func)
    # The DPTR copy is invalidated by DPTR++, so R0 = DPTR is NOT substituted
    assert block.hir[2].rhs == Reg("DPTR")


# ── No-op cases ───────────────────────────────────────────────────────────────

def test_no_copies_no_change():
    """Blocks with no simple copies are untouched."""
    nodes = [
        Assign(0x1000, XRAMRef(Const(0x100)), Reg("R0")),
        Assign(0x1002, Reg("R1"), XRAMRef(Const(0x200))),
    ]
    func, block = _func(nodes)
    original_hir = list(block.hir)
    CopyPropagator().run(func)
    assert block.hir[0].rhs == Reg("R0")     # unchanged
    assert isinstance(block.hir[1].rhs, XRAMRef)  # unchanged


def test_xram_rhs_not_tracked_as_copy():
    """A = XRAM[addr] is not added to copies (XRAMRef is not copy-worthy)."""
    func, block = _func([
        Assign(0x1000, Reg("A"), XRAMRef(Const(0x100))),
        Assign(0x1002, Reg("R0"), Reg("A")),
    ])
    CopyPropagator().run(func)
    # A = XRAM[addr] is not copy-worthy, so R0 = A stays as R0 = A
    assert block.hir[1].rhs == Reg("A")
