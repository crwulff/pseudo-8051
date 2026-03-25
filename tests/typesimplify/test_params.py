from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param
from pseudo8051.ir.hir import Statement

from ..helpers import make_single_block_func


def _texts(func):
    return [n.text for n in func.hir if isinstance(n, Statement)]


class TestSingleRegParam:
    def test_basic_accum_relay_with_param(self):
        """A = R7; XRAM[X] = A; with proto count→R7 → XRAM[X] = count;"""
        PROTOTYPES["f"] = FuncProto(
            return_type="void",
            params=[Param("count", "uint8_t", ("R7",))],
        )
        func = make_single_block_func("f", ["A = R7;", "XRAM[X] = A;"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert texts == ["XRAM[X] = count;"]

    def test_no_proto_structural_patterns_run(self):
        """Without a prototype, structural patterns (AccumRelayPattern) still fire."""
        func = make_single_block_func("unknown_fn", ["A = R7;", "XRAM[X] = A;"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert "XRAM[X] = R7;" in texts
        assert "A = R7;" not in texts
        assert "XRAM[X] = A;" not in texts

    def test_return_statement_preserved(self):
        """return; statement passes through unchanged."""
        PROTOTYPES["g"] = FuncProto(
            return_type="void",
            params=[Param("val", "uint8_t", ("R7",))],
        )
        func = make_single_block_func("g", ["A = R7;", "XRAM[X] = A;", "return;"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert "return;" in texts


class TestUsercallParamSubst:
    def test_three_relay_pairs(self):
        """Three A-relay pairs with params in R7, R5, R3 → all substituted."""
        PROTOTYPES["h"] = FuncProto(
            return_type="void",
            params=[
                Param("H", "uint8_t", ("R7",)),
                Param("M", "uint8_t", ("R5",)),
                Param("L", "uint8_t", ("R3",)),
            ],
        )
        func = make_single_block_func("h", [
            "A = R7;", "XRAM[X1] = A;",
            "A = R5;", "XRAM[X2] = A;",
            "A = R3;", "XRAM[X3] = A;",
        ])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert "XRAM[X1] = H;" in texts
        assert "XRAM[X2] = M;" in texts
        assert "XRAM[X3] = L;" in texts

    def test_pair_param(self):
        """16-bit param in R6R7: pair substituted in expression."""
        PROTOTYPES["p"] = FuncProto(
            return_type="void",
            params=[Param("val", "uint16_t", ("R6", "R7"))],
        )
        func = make_single_block_func("p", ["XRAM[X] = R6R7;"])
        TypeAwareSimplifier().run(func)
        texts = _texts(func)
        assert "XRAM[X] = val;" in texts
