from pseudo8051.ir.hir import Statement
from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param

from ..helpers import make_single_block_func


class TestEndToEndPipeline:

    def test_accum_relay_with_param(self):
        """
        Single block: ["A = R7;", "XRAM[PORT] = A;", "return;"]
        Proto: void f(uint8_t H) with H in R7.
        Expected hir after TypeAwareSimplifier: ["XRAM[PORT] = H;", "return;"]
        """
        PROTOTYPES["f"] = FuncProto(
            return_type="void",
            params=[Param("H", "uint8_t", ("R7",))],
        )
        func = make_single_block_func("f", [
            "A = R7;", "XRAM[PORT] = A;", "return;",
        ])
        TypeAwareSimplifier().run(func)

        texts = [n.text for n in func.hir if isinstance(n, Statement)]
        assert "XRAM[PORT] = H;" in texts
        assert "return;" in texts
        assert "A = R7;" not in texts
        assert "XRAM[PORT] = A;" not in texts

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
            "R4 = 0x00;", "R5 = 0x00;", "R6 = 0x5d;", "R7 = 0xc0;",
            "return div32(R4R5R6R7, R0R1R2R3);",
        ])
        TypeAwareSimplifier().run(func)

        texts = [n.text for n in func.hir if isinstance(n, Statement)]
        assert any("0x00005dc0" in t for t in texts), \
            f"Expected 0x00005dc0 in output, got: {texts}"

    def test_neg16_in_pipeline(self):
        """7-statement SUBB negation in a single-block function → x = -x;"""
        PROTOTYPES["neg_fn"] = FuncProto(
            return_type="void",
            params=[Param("x", "int16_t", ("R6", "R7"))],
        )
        func = make_single_block_func("neg_fn", [
            "C = 0;",
            "A = 0;", "A -= R7 + C;", "R7 = A;",
            "A = 0;", "A -= R6 + C;", "R6 = A;",
        ])
        TypeAwareSimplifier().run(func)

        texts = [n.text for n in func.hir if isinstance(n, Statement)]
        assert "x = -x;" in texts
