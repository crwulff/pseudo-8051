from pseudo8051.ir.hir import Assign, CompoundAssign, ReturnStmt
from pseudo8051.ir.expr import Reg, Const, Name, XRAMRef, BinOp, Call
from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param

from ..helpers import make_single_block_func


class TestEndToEndPipeline:

    def test_accum_relay_with_param(self):
        """
        Single block: A=R7; XRAM[PORT]=A; return;
        Proto: void f(uint8_t H) with H in R7.
        Expected hir after TypeAwareSimplifier: ["XRAM[PORT] = H;", "return;"]
        """
        PROTOTYPES["f"] = FuncProto(
            return_type="void",
            params=[Param("H", "uint8_t", ("R7",))],
        )
        func = make_single_block_func("f", [
            Assign(0x1000, Reg("A"), Reg("R7")),
            Assign(0x1002, XRAMRef(Name("PORT")), Reg("A")),
            ReturnStmt(0x1004, None),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        assert "XRAM[PORT] = H;" in rendered
        assert "return;" in rendered
        assert "A = R7;" not in rendered
        assert "XRAM[PORT] = A;" not in rendered

    def test_const_group_no_proto(self):
        """
        No caller proto but callee has a prototype with R4R5R6R7 params.
        Direct register const-load + call → constant folded into call args.
        """
        PROTOTYPES["div32"] = FuncProto(
            return_type="uint32_t",
            return_regs=("R4", "R5", "R6", "R7"),
            params=[
                Param("dividend", "uint32_t", ("R4", "R5", "R6", "R7")),
                Param("divisor",  "uint32_t", ("R0", "R1", "R2", "R3")),
            ],
        )
        func = make_single_block_func("caller", [
            Assign(0x1000, Reg("R4"), Const(0x00)),
            Assign(0x1002, Reg("R5"), Const(0x00)),
            Assign(0x1004, Reg("R6"), Const(0x5d)),
            Assign(0x1006, Reg("R7"), Const(0xc0)),
            ReturnStmt(0x1008, Call("div32", [Name("R4R5R6R7"), Name("R0R1R2R3")])),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        assert any("0x00005dc0" in t for t in rendered), \
            f"Expected 0x00005dc0 in output, got: {rendered}"

    def test_neg16_in_pipeline(self):
        """7-statement SUBB negation in a single-block function → x = -x;"""
        PROTOTYPES["neg_fn"] = FuncProto(
            return_type="void",
            params=[Param("x", "int16_t", ("R6", "R7"))],
        )
        func = make_single_block_func("neg_fn", [
            Assign(0x1000, Reg("C"), Const(0)),
            Assign(0x1002, Reg("A"), Const(0)),
            CompoundAssign(0x1004, Reg("A"), "-=", BinOp(Reg("R7"), "+", Reg("C"))),
            Assign(0x1006, Reg("R7"), Reg("A")),
            Assign(0x1008, Reg("A"), Const(0)),
            CompoundAssign(0x100a, Reg("A"), "-=", BinOp(Reg("R6"), "+", Reg("C"))),
            Assign(0x100c, Reg("R6"), Reg("A")),
        ])
        TypeAwareSimplifier().run(func)

        rendered = [t for n in func.hir for _, t in n.render()]
        assert "x = -x;" in rendered
