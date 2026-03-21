"""
tests/test_pipeline.py — Structural pass and end-to-end pipeline tests.

Uses FakeBlock / FakeFunction from helpers.py.  Passes are instantiated and
called directly; no run_all_passes() (which needs full IDA context).
"""

import pytest

from pseudo8051.ir.hir import Statement, ForNode, WhileNode, IfNode
from pseudo8051.passes.loops  import LoopStructurer
from pseudo8051.passes.ifelse import IfElseStructurer
from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param

from .helpers import FakeBlock, FakeFunction, connect, make_single_block_func


# ── Fixture: clean PROTOTYPES between tests ───────────────────────────────────

@pytest.fixture(autouse=True)
def clean_prototypes():
    keys_before = set(PROTOTYPES.keys())
    yield
    for k in list(PROTOTYPES.keys()):
        if k not in keys_before:
            del PROTOTYPES[k]


# ── LoopStructurer tests ──────────────────────────────────────────────────────

class TestLoopStructurer:

    def test_djnz_for_loop(self):
        """
        DJNZ self-loop with init block → ForNode.

        Topology:
            init  (0x1000):  "R7 = 5;"   → header
            header(0x1010):  (empty)      → body
            body  (0x1018):  "XRAM[S] = A;"
                             "if (--R7 != 0) goto label_1010;"
                             → [header (back-edge), after]
            after (0x1020):  (empty)
        """
        init   = FakeBlock(0x1000, hir=[Statement(0x1000, "R7 = 5;")])
        header = FakeBlock(0x1010, hir=[])
        body   = FakeBlock(0x1018, hir=[
            Statement(0x1018, "XRAM[S] = A;"),
            Statement(0x101a, "if (--R7 != 0) goto label_1010;"),
        ])
        after  = FakeBlock(0x1020, hir=[])

        connect(init,   header)
        connect(header, body)
        connect(body,   header)   # back-edge
        connect(body,   after)

        func = FakeFunction("djnz_f", [init, header, body, after])
        LoopStructurer().run(func)

        # header.hir should now be [ForNode]
        assert len(header.hir) == 1
        fn = header.hir[0]
        assert isinstance(fn, ForNode)
        assert fn.init      == "R7 = 5"
        assert fn.condition == "R7"
        assert fn.update    == "--R7"
        assert len(fn.body_nodes) == 1
        assert isinstance(fn.body_nodes[0], Statement)
        assert fn.body_nodes[0].text == "XRAM[S] = A;"

        # body block absorbed
        assert body._absorbed

    def test_djnz_no_init_produces_while(self):
        """DJNZ loop without a detectable init value → WhileNode."""
        header = FakeBlock(0x1000, hir=[])
        body   = FakeBlock(0x1008, hir=[
            Statement(0x1008, "if (--R7 != 0) goto label_1000;"),
        ])
        after  = FakeBlock(0x1010, hir=[])

        connect(header, body)
        connect(body,   header)   # back-edge
        connect(body,   after)

        func = FakeFunction("djnz_noloop", [header, body, after])
        LoopStructurer().run(func)

        assert len(header.hir) == 1
        wn = header.hir[0]
        assert isinstance(wn, WhileNode)
        assert wn.condition == "--R7 != 0"

    def test_while_self_loop(self):
        """
        Conditional self-loop (do-while style) → WhileNode with body.

        Topology:
            header (0x1000): "XRAM[S] = A;"
                             "if (C != 0) goto label_1000;"
                             → [header (back-edge), after]
            after  (0x1010): (empty)
        """
        header = FakeBlock(0x1000, hir=[
            Statement(0x1000, "XRAM[S] = A;"),
            Statement(0x1002, "if (C != 0) goto label_1000;"),
        ])
        after  = FakeBlock(0x1010, hir=[])

        connect(header, header)   # self-loop (back-edge)
        connect(header, after)

        func = FakeFunction("while_f", [header, after])
        LoopStructurer().run(func)

        assert len(header.hir) == 1
        wn = header.hir[0]
        assert isinstance(wn, WhileNode)
        assert wn.condition == "C != 0"
        assert len(wn.body_nodes) == 1
        assert wn.body_nodes[0].text == "XRAM[S] = A;"

    def test_no_loop_no_change(self):
        """Linear CFG without back-edges is left unchanged."""
        b0 = FakeBlock(0x1000, hir=[Statement(0x1000, "A = R7;")])
        b1 = FakeBlock(0x1010, hir=[Statement(0x1010, "return;")])
        connect(b0, b1)
        func = FakeFunction("linear", [b0, b1])
        LoopStructurer().run(func)
        assert b0.hir[0].text == "A = R7;"
        assert b1.hir[0].text == "return;"


# ── IfElseStructurer tests ────────────────────────────────────────────────────

class TestIfElseStructurer:

    def test_if_then_no_else(self):
        """
        Block A → [B (true), C (false/merge)], B → C.
        Result: A.hir = [IfNode(cond, then=[...], else=[])]

        Topology:
            A (0x1000): "if (A != 0) goto label_b;" → [B, C]
            B (0x1010): "R7 = 0;"                   → [C]
            C (0x1020): "return;"                    (merge)
        """
        A = FakeBlock(0x1000, hir=[Statement(0x1000, "if (A != 0) goto label_b;")])
        B = FakeBlock(0x1010, hir=[Statement(0x1010, "R7 = 0;")], label="label_b")
        C = FakeBlock(0x1020, hir=[Statement(0x1020, "return;")])

        connect(A, B)
        connect(A, C)
        connect(B, C)

        func = FakeFunction("if_f", [A, B, C])
        IfElseStructurer().run(func)

        assert len(A.hir) == 1
        node = A.hir[0]
        assert isinstance(node, IfNode)
        assert node.condition == "A != 0"
        assert len(node.then_nodes) == 1
        assert node.then_nodes[0].text == "R7 = 0;"
        assert node.else_nodes == []
        assert B._absorbed

    def test_if_else(self):
        """
        Block A → [B (true), D (false)], both converge at E.

        Topology:
            A (0x1000): "if (A != 0) goto label_b;" → [B, D]
            B (0x1010): "R7 = 1;"  → [E]   (label_b)
            D (0x1020): "R7 = 2;"  → [E]
            E (0x1030): "return;"
        """
        A = FakeBlock(0x1000, hir=[Statement(0x1000, "if (A != 0) goto label_b;")])
        B = FakeBlock(0x1010, hir=[Statement(0x1010, "R7 = 1;")], label="label_b")
        D = FakeBlock(0x1020, hir=[Statement(0x1020, "R7 = 2;")])
        E = FakeBlock(0x1030, hir=[Statement(0x1030, "return;")])

        connect(A, B)
        connect(A, D)
        connect(B, E)
        connect(D, E)

        func = FakeFunction("ifelse_f", [A, B, D, E])
        IfElseStructurer().run(func)

        node = A.hir[0]
        assert isinstance(node, IfNode)
        assert node.condition == "A != 0"
        assert node.then_nodes[0].text == "R7 = 1;"
        assert node.else_nodes[0].text == "R7 = 2;"

    def test_no_structure_single_successor(self):
        """Block with only one successor is not structured."""
        A = FakeBlock(0x1000, hir=[Statement(0x1000, "A = R7;")])
        B = FakeBlock(0x1010, hir=[Statement(0x1010, "return;")])
        connect(A, B)
        func = FakeFunction("linear_if", [A, B])
        IfElseStructurer().run(func)
        assert isinstance(A.hir[0], Statement)

    def test_empty_then_arm_swapped(self):
        """
        When true arm is empty (jumps straight to merge), arms are swapped
        and condition is inverted so then_nodes is always non-empty.

        Topology:
            A (0x1000): "if (A != 0) goto label_c;" → [C (true=merge), B (false)]
            B (0x1010): "R7 = 0;" → [C]
            C (0x1020): "return;"
        """
        A = FakeBlock(0x1000, hir=[Statement(0x1000, "if (A != 0) goto label_c;")])
        B = FakeBlock(0x1010, hir=[Statement(0x1010, "R7 = 0;")])
        C = FakeBlock(0x1020, hir=[Statement(0x1020, "return;")], label="label_c")

        connect(A, C)   # true arm (goto label_c) → directly to merge
        connect(A, B)   # false arm
        connect(B, C)

        func = FakeFunction("swap_f", [A, B, C])
        IfElseStructurer().run(func)

        node = A.hir[0]
        assert isinstance(node, IfNode)
        # Condition should be inverted: !(A != 0)
        assert "A != 0" in node.condition
        assert node.then_nodes[0].text == "R7 = 0;"
        assert node.else_nodes == []


# ── End-to-end TypeAwareSimplifier pipeline test ─────────────────────────────

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

        Note: loads must be direct (R4 = imm), not via A-relay.  AccumRelayPattern
        runs before ConstGroupPattern; using A as a carrier would trigger the relay
        pattern first, breaking the const-group sequence.
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
        # The constant group folds into the call
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
