"""
tests/pipeline/test_post_inc_load.py — Integration tests for RegPostIncPattern and RegPreIncPattern.

Verifies that the full pipeline (TypeAwareSimplifier + _propagate_values) correctly
produces IRAM[dst] = IRAM[src.lo++] from the 8051 indirect-with-post-increment idiom:

    mov  A, @R1     →  A = IRAM[R1]
    inc  R1         →  R1++
    mov  @R0, A     →  IRAM[R0] = A
"""

from pseudo8051.ir.hir import Assign, ExprStmt
from pseudo8051.ir.expr import Reg, UnaryOp, IRAMRef, XRAMRef, Const
from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param

from ..helpers import make_single_block_func


class TestPostIncLoadPipeline:

    def test_iram_post_inc_folded_into_write(self):
        """
        mov A, @R1; inc R1; mov @R0, A  with src@<R2:R1>, dest@<R4:R0> proto
        → IRAM[dest.lo] = IRAM[src.lo++];
        """
        PROTOTYPES["pil_iram_write"] = FuncProto(
            return_type="void",
            params=[
                Param("src",  "uint16_t", ("R2", "R1")),
                Param("dest", "uint16_t", ("R4", "R0")),
            ],
        )
        func = make_single_block_func("pil_iram_write", [
            Assign(0x1000, Reg("A"),          IRAMRef(Reg("R1"))),
            ExprStmt(0x1001, UnaryOp("++", Reg("R1"), post=True)),
            Assign(0x1002, IRAMRef(Reg("R0")), Reg("A")),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        assert "IRAM[dest.lo] = IRAM[src.lo++];" in rendered, \
            f"Expected post-inc fold with dest.lo, got: {rendered}"
        # The intermediate A = IRAM[R1] node must be gone.
        assert all("= A;" not in t for t in rendered), \
            f"Intermediate A assignment should be eliminated, got: {rendered}"

    def test_xram_post_inc_folded_into_write(self):
        """
        movx A, @R1; inc R1; movx @R0, A  with src@<R2:R1>, dest@<R4:R0> proto
        → XRAM[dest.lo] = XRAM[src.lo++];
        """
        PROTOTYPES["pil_xram_write"] = FuncProto(
            return_type="void",
            params=[
                Param("src",  "uint16_t", ("R2", "R1")),
                Param("dest", "uint16_t", ("R4", "R0")),
            ],
        )
        func = make_single_block_func("pil_xram_write", [
            Assign(0x1000, Reg("A"),           XRAMRef(Reg("R1"))),
            ExprStmt(0x1001, UnaryOp("++", Reg("R1"), post=True)),
            Assign(0x1002, XRAMRef(Reg("R0")), Reg("A")),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        assert "XRAM[dest.lo] = XRAM[src.lo++];" in rendered, \
            f"Expected post-inc fold with dest.lo, got: {rendered}"

    def test_iram_post_inc_store(self):
        """
        mov @R0, A; inc R0  with dest@<R4:R0> proto
        → IRAM[dest.lo++] = ...;
        """
        PROTOTYPES["pil_iram_store"] = FuncProto(
            return_type="void",
            params=[Param("dest", "uint16_t", ("R4", "R0"))],
        )
        func = make_single_block_func("pil_iram_store", [
            Assign(0x1000, IRAMRef(Reg("R0")), Reg("A")),
            ExprStmt(0x1001, UnaryOp("++", Reg("R0"), post=True)),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        assert "IRAM[dest.lo++] = A;" in rendered, \
            f"Expected post-inc store fold, got: {rendered}"
        assert all("dest.lo++;" not in t for t in rendered), \
            f"Separate dest.lo++ should be gone, got: {rendered}"

    def test_load_and_store_post_inc(self):
        """
        mov A, @R1; inc R1; mov @R0, A; inc R0  with src/dest protos
        → IRAM[dest.lo++] = IRAM[src.lo++];
        """
        PROTOTYPES["pil_copy"] = FuncProto(
            return_type="void",
            params=[
                Param("src",  "uint16_t", ("R2", "R1")),
                Param("dest", "uint16_t", ("R4", "R0")),
            ],
        )
        func = make_single_block_func("pil_copy", [
            Assign(0x1000, Reg("A"),          IRAMRef(Reg("R1"))),
            ExprStmt(0x1001, UnaryOp("++", Reg("R1"), post=True)),
            Assign(0x1002, IRAMRef(Reg("R0")), Reg("A")),
            ExprStmt(0x1003, UnaryOp("++", Reg("R0"), post=True)),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        assert "IRAM[dest.lo++] = IRAM[src.lo++];" in rendered, \
            f"Expected combined load+store post-inc fold, got: {rendered}"
        assert len([t for t in rendered if t.strip()]) == 1, \
            f"Expected single output line, got: {rendered}"

    def test_no_fold_without_inc(self):
        """Without the Rn++ node, AccumRelayPattern folds to IRAM[dest.lo] = IRAM[src.lo]."""
        PROTOTYPES["pil_no_inc"] = FuncProto(
            return_type="void",
            params=[
                Param("src",  "uint16_t", ("R2", "R1")),
                Param("dest", "uint16_t", ("R4", "R0")),
            ],
        )
        func = make_single_block_func("pil_no_inc", [
            Assign(0x1000, Reg("A"),          IRAMRef(Reg("R1"))),
            Assign(0x1001, IRAMRef(Reg("R0")), Reg("A")),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        # AccumRelayPattern folds A relay + LHS is now renamed via _transform_default fix.
        assert "IRAM[dest.lo] = IRAM[src.lo];" in rendered, \
            f"Expected simple relay fold with named LHS, got: {rendered}"
        assert all("++" not in t for t in rendered), \
            f"Unexpected ++ in output: {rendered}"

    def test_dptr_post_inc_folded(self):
        """
        movx A, @DPTR; inc DPTR  — DPTR is now included in RegPostIncPattern.
        The standalone DPTR++ is eliminated; it is embedded as XRAM[DPTR++].
        Note: full 3-node fusion into IRAM[dest.lo] = XRAM[DPTR++] requires
        _propagate_values to inline, which it cannot do for DPTR (Reg stays Reg,
        not substituted to a Name). Two nodes remain instead of three.
        """
        PROTOTYPES["pil_dptr_load"] = FuncProto(
            return_type="void",
            params=[
                Param("ptr",  "uint16_t", ("DPH", "DPL")),
                Param("dest", "uint16_t", ("R4", "R0")),
            ],
        )
        func = make_single_block_func("pil_dptr_load", [
            Assign(0x1000, Reg("A"),           XRAMRef(Reg("DPTR"))),
            ExprStmt(0x1001, UnaryOp("++", Reg("DPTR"), post=True)),
            Assign(0x1002, IRAMRef(Reg("R0")), Reg("A")),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        # The standalone DPTR++ ExprStmt must be gone (folded into the load)
        assert any("XRAM[DPTR++]" in t for t in rendered), \
            f"Expected DPTR++ embedded in XRAM access, got: {rendered}"
        assert all("DPTR++;" not in t for t in rendered), \
            f"Standalone DPTR++ should be eliminated, got: {rendered}"
        # Three nodes collapsed to two
        assert len([t for t in rendered if t.strip()]) == 2, \
            f"Expected 2 output lines (3-to-2 fold), got: {rendered}"

    def test_pre_inc_iram_load(self):
        """
        inc R1; mov A, @R1  with src@<R2:R1> proto
        → A = IRAM[++src.lo];  (RegPreIncPattern embeds pre-inc)
        """
        PROTOTYPES["pil_pre_inc"] = FuncProto(
            return_type="void",
            params=[Param("src", "uint16_t", ("R2", "R1"))],
        )
        func = make_single_block_func("pil_pre_inc", [
            ExprStmt(0x1000, UnaryOp("++", Reg("R1"), post=True)),
            Assign(0x1001, Reg("A"), IRAMRef(Reg("R1"))),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        assert "A = IRAM[++src.lo];" in rendered, \
            f"Expected pre-inc fold, got: {rendered}"
        assert all("src.lo++;" not in t for t in rendered), \
            f"Standalone src.lo++ should be gone, got: {rendered}"
