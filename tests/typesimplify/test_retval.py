from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param
from pseudo8051.ir.hir import Statement

from ..helpers import make_single_block_func


def _texts(func):
    return [n.text for n in func.hir if isinstance(n, Statement)]


class TestRetvalRenaming:
    def test_u32_retval(self):
        """R4R5R6R7 = callee(R6R7); → uint32_t retval1 = callee(arg);"""
        PROTOTYPES["callee"] = FuncProto(
            return_type="uint32_t",
            return_regs=("R4", "R5", "R6", "R7"),
            params=[Param("arg", "uint16_t", ("R6", "R7"))],
        )
        func = make_single_block_func("caller", ["R4R5R6R7 = callee(R6R7);"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert len(texts) == 1
        assert texts[0] == "uint32_t retval1 = callee(arg);"

    def test_u8_retval(self):
        """A = getter(); with no caller proto → A stays as-is (A excluded from callee regmap)."""
        PROTOTYPES["getter"] = FuncProto(
            return_type="uint8_t",
            return_regs=("A",),
            params=[],
        )
        func = make_single_block_func("user", ["A = getter();"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert texts[0] == "A = getter();"
