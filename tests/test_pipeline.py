"""
tests/test_pipeline.py — Structural pass and end-to-end pipeline tests.

Uses FakeBlock / FakeFunction from helpers.py.  Passes are instantiated and
called directly; no run_all_passes() (which needs full IDA context).
"""

import pytest

from pseudo8051.ir.hir  import Statement, ForNode, WhileNode, IfNode, IfGoto, Assign, CompoundAssign, SwitchNode
from pseudo8051.ir.expr import Reg, Const, BinOp, UnaryOp, Name, XRAMRef
from pseudo8051.passes.loops  import LoopStructurer
from pseudo8051.passes.ifelse import IfElseStructurer
from pseudo8051.passes.switch import SwitchStructurer
from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param

from .helpers import FakeBlock, FakeFunction, connect, make_single_block_func


def _cond_str(c) -> str:
    """Render a condition that may be a str or Expr."""
    return c.render() if hasattr(c, "render") else c


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
        assert fn.init == "R7 = 5"
        assert _cond_str(fn.condition) == "R7"
        assert _cond_str(fn.update)    == "--R7"
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
        assert _cond_str(wn.condition) == "--R7 != 0"

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

    def test_dead_end_false_arm(self):
        """
        False arm is a terminal block (no successors) — the 'if-with-exit' pattern.
        _is_dead_end(false_block) == True  →  merge_ea = true_block.start_ea,
        arms swapped, condition inverted.

        Topology:
            A (0x1000): "if (A != 0) goto label_b;" → [B (true), C (false)]
            C (0x1010): "return;"                    (dead-end, no successors)
            B (0x1020): "R7 = 1;"                   (continuation, label_b)
        """
        A = FakeBlock(0x1000, hir=[Statement(0x1000, "if (A != 0) goto label_b;")])
        C = FakeBlock(0x1010, hir=[Statement(0x1010, "return;")])
        B = FakeBlock(0x1020, hir=[Statement(0x1020, "R7 = 1;")], label="label_b")

        connect(A, B)   # true arm
        connect(A, C)   # false arm (dead-end)
        # C has no successors

        func = FakeFunction("dead_end_f", [A, C, B])
        IfElseStructurer().run(func)

        node = A.hir[0]
        assert isinstance(node, IfNode)
        # Condition inverted because then was empty (true arm = merge = empty)
        assert "A != 0" in (node.condition if isinstance(node.condition, str)
                            else node.condition.render())
        assert len(node.then_nodes) == 1
        assert node.then_nodes[0].text == "return;"
        assert node.else_nodes == []
        assert C._absorbed

    def test_dead_end_false_arm_multiblock(self):
        """
        False arm spans two blocks before terminating — multi-block dead-end path.
        _is_dead_end(false_block) == True  →  merge_ea = true_block.start_ea,
        both dead-end blocks absorbed into then_nodes after arm swap.

        Topology:
            A (0x1000): "if (A != 0) goto label_b;" → [B (true), C (false)]
            C (0x1010): "R6 = 0;"                   → [D]
            D (0x1020): "return;"                    (no successors)
            B (0x1030): "R7 = 1;"                   (continuation, label_b)
        """
        A = FakeBlock(0x1000, hir=[Statement(0x1000, "if (A != 0) goto label_b;")])
        C = FakeBlock(0x1010, hir=[Statement(0x1010, "R6 = 0;")])
        D = FakeBlock(0x1020, hir=[Statement(0x1020, "return;")])
        B = FakeBlock(0x1030, hir=[Statement(0x1030, "R7 = 1;")], label="label_b")

        connect(A, B)   # true arm
        connect(A, C)   # false arm (dead-end path: C → D, D has no successors)
        connect(C, D)

        func = FakeFunction("dead_end_multi_f", [A, C, D, B])
        IfElseStructurer().run(func)

        node = A.hir[0]
        assert isinstance(node, IfNode)
        assert "A != 0" in (node.condition if isinstance(node.condition, str)
                            else node.condition.render())
        texts = [n.text for n in node.then_nodes]
        assert "R6 = 0;" in texts
        assert "return;" in texts
        assert node.else_nodes == []
        assert C._absorbed
        assert D._absorbed

    def test_arm_goto_stripped_with_ida_label(self):
        """
        Else arm ends with 'goto code_7_2cfa;' to the merge block.
        When the merge block has an IDA-assigned label (not label_XXXX format),
        the goto must still be stripped from the arm HIR.

        Topology:
            A (0x1000): "if (A != 0) goto code_7_2cf8;" → [C (true), B (false)]
            B (0x1010): "R7 = 1;" + "goto code_7_2cfa;"  → [E]
            C (0x1020): "R7 = 0;"  (label code_7_2cf8)   → [E]
            E (0x1030): "A = R7;"  (label code_7_2cfa)    (merge)
        """
        A = FakeBlock(0x1000, hir=[Statement(0x1000,
                                             "if (A != 0) goto code_7_2cf8;")])
        B = FakeBlock(0x1010, hir=[Statement(0x1010, "R7 = 1;"),
                                   Statement(0x1012, "goto code_7_2cfa;")])
        C = FakeBlock(0x1020, hir=[Statement(0x1020, "R7 = 0;")],
                      label="code_7_2cf8")
        E = FakeBlock(0x1030, hir=[Statement(0x1030, "A = R7;")],
                      label="code_7_2cfa")

        connect(A, C)   # true arm
        connect(A, B)   # false arm
        connect(B, E)
        connect(C, E)

        func = FakeFunction("ida_label_f", [A, B, C, E])
        IfElseStructurer().run(func)

        node = A.hir[0]
        assert isinstance(node, IfNode)
        # else body: R7=1 only — goto code_7_2cfa must have been stripped
        assert len(node.else_nodes) == 1
        assert node.else_nodes[0].text == "R7 = 1;"

    def test_dead_end_arm_skipped_if_externally_referenced(self):
        """
        E's dead-end false arm (F, label 'label_f') is also the target of A's
        goto.  E must NOT be structured (F cannot be absorbed), but A can be
        structured using F as the merge point.

        Topology:
            A (0x1000): "if (X == 0) goto label_f;"       → [F (true), B (false)]
            B (0x1010): "R7 = 1;"                          → [E]
            E (0x1020): "if (R7 != 0) goto label_g;"      → [G (true), F (false)]
            F (0x1030): "return;"   (label_f, no succ.)    — dead-end, external ref
            G (0x1040): "R7 = 2;"
        """
        A = FakeBlock(0x1000, hir=[Statement(0x1000, "if (X == 0) goto label_f;")])
        B = FakeBlock(0x1010, hir=[Statement(0x1010, "R7 = 1;")])
        E = FakeBlock(0x1020, hir=[Statement(0x1020, "if (R7 != 0) goto label_g;")])
        F = FakeBlock(0x1030, hir=[Statement(0x1030, "return;")], label="label_f")
        G = FakeBlock(0x1040, hir=[Statement(0x1040, "R7 = 2;")])

        connect(A, F)   # true arm (goto label_f)
        connect(A, B)   # false arm
        connect(B, E)
        connect(E, G)   # true arm (goto label_g)
        connect(E, F)   # false arm (dead-end fall-through)

        func = FakeFunction("ext_ref_f", [A, B, E, F, G])
        IfElseStructurer().run(func)

        # F must NOT have been absorbed (its label is externally referenced by A)
        assert not F._absorbed
        # A's false arm [B, E] has no externally-referenced labels → A is structured
        assert B._absorbed
        assert E._absorbed


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


# ── AccumFoldPattern tests ────────────────────────────────────────────────────

class TestAccumFoldPattern:
    """Test AccumFoldPattern directly (bypasses TypeAwareSimplifier reg_map guard)."""

    def _match(self, nodes):
        from pseudo8051.passes.patterns.accum_fold import AccumFoldPattern
        return AccumFoldPattern().match(nodes, 0, {}, lambda ns, rm: ns)

    def test_xram_load_mask_ifgoto(self):
        """
        DPTR=sym; A=XRAM[sym]; A&=1; IfGoto(A==0, label)
        → IfGoto((XRAM[sym] & 1) == 0, label)
        DPTR assignment consumed, 1 compound assign.
        """
        nodes = [
            Assign(0x1000, Reg("DPTR"), Name("EXT_28")),
            Assign(0x1002, Reg("A"),    XRAMRef(Name("EXT_28"))),
            CompoundAssign(0x1004, Reg("A"), "&=", Const(1)),
            IfGoto(0x1006, BinOp(Reg("A"), "==", Const(0)), "label_1010"),
        ]
        result = self._match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 4
        assert len(replacement) == 1
        node = replacement[0]
        assert isinstance(node, IfGoto)
        assert node.label == "label_1010"
        assert node.cond.render() == "(XRAM[EXT_28] & 1) == 0"

    def test_load_mask_ifnode(self):
        """
        A=XRAM[sym]; A&=1; IfNode(A!=0, [then_stmt])
        → IfNode((XRAM[sym] & 1) != 0, [then_stmt])
        No DPTR prefix, structured if.
        """
        then_stmt = Statement(0x1010, "R7 = 1;")
        nodes = [
            Assign(0x1000, Reg("A"), XRAMRef(Name("SYM"))),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(1)),
            IfNode(0x1004, BinOp(Reg("A"), "!=", Const(0)), [then_stmt]),
        ]
        result = self._match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 3
        node = replacement[0]
        assert isinstance(node, IfNode)
        assert isinstance(node.condition, BinOp)
        assert node.condition.render() == "(XRAM[SYM] & 1) != 0"
        assert len(node.then_nodes) == 1

    def test_reg_load_mask_relay(self):
        """
        A=R7; A&=0xf0; Assign(R6, A)
        → Assign(R6, R7 & 0xf0)
        Register source, compound assign, store relay.
        """
        nodes = [
            Assign(0x1000, Reg("A"),  Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "&=", Const(0xf0)),
            Assign(0x1004, Reg("R6"), Reg("A")),
        ]
        result = self._match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 3
        node = replacement[0]
        assert isinstance(node, Assign)
        assert node.lhs == Reg("R6")
        assert node.rhs.render() == "R7 & 0xf0"

    def test_ifgoto_no_compound(self):
        """
        A=XRAM[sym]; IfGoto(A!=0, label)
        → IfGoto(XRAM[sym] != 0, label)
        0 compound assigns, IfGoto terminal — verifies this case is NOT left to
        AccumRelayPattern (which wouldn't handle IfGoto).
        """
        nodes = [
            Assign(0x1000, Reg("A"), XRAMRef(Name("SYM"))),
            IfGoto(0x1002, BinOp(Reg("A"), "!=", Const(0)), "label_1020"),
        ]
        result = self._match(nodes)
        assert result is not None
        replacement, new_i = result
        assert new_i == 2
        node = replacement[0]
        assert isinstance(node, IfGoto)
        assert node.label == "label_1020"
        assert node.cond.render() == "XRAM[SYM] != 0"

    def test_pure_relay_unaffected(self):
        """
        A=XRAM[sym]; Assign(R7, A) — no compound assigns, no DPTR prefix.
        AccumFoldPattern must return None; AccumRelayPattern owns this case.
        """
        nodes = [
            Assign(0x1000, Reg("A"),  XRAMRef(Name("SYM"))),
            Assign(0x1002, Reg("R7"), Reg("A")),
        ]
        result = self._match(nodes)
        assert result is None


# ── SwitchStructurer tests ────────────────────────────────────────────────────

class TestSwitchStructurer:
    """Tests for the SwitchStructurer pass."""

    def test_switch_basic_jz_chain(self):
        """
        3-block jz chain → SwitchNode with 2 cases, no default.

        Topology:
            b0 (0x1000): A=R7; A+=0xFE(-2); jz label_c2 → [b_c2, b1]
            b1 (0x1010): A+=0xFE(-2);        jz label_c4 → [b_c4, b_fall]
            b_c2 (0x1020): label_c2 (case target)
            b_c4 (0x1030): label_c4 (case target)
            b_fall (0x1040): (fall-through after switch)

        Case values: cumulative after b0 = 0xFE → case 2;
                     cumulative after b1 = 0xFC → case 4.
        """
        b_c2   = FakeBlock(0x1020, hir=[Statement(0x1020, "return;")], label="label_c2")
        b_c4   = FakeBlock(0x1030, hir=[Statement(0x1030, "return;")], label="label_c4")
        b_fall = FakeBlock(0x1040, hir=[Statement(0x1040, "return;")])
        b1     = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1012, BinOp(Reg("A"), "==", Const(0)), "label_c4"),
        ])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2)
        connect(b0, b1)
        connect(b1, b_c4)
        connect(b1, b_fall)

        func = FakeFunction("sw_basic", [b0, b1, b_c2, b_c4, b_fall])
        SwitchStructurer().run(func)

        assert len(b0.hir) == 1
        sw = b0.hir[0]
        assert isinstance(sw, SwitchNode)
        assert sw.subject == Reg("R7")
        assert sw.default_label is None
        # Two case entries
        assert len(sw.cases) == 2
        all_cases = {v: lbl for vals, lbl in sw.cases for v in vals}
        assert all_cases[2] == "label_c2"
        assert all_cases[4] == "label_c4"
        # b1 absorbed; case targets and fall-through left alone
        assert b1._absorbed
        assert not b_c2._absorbed
        assert not b_c4._absorbed

    def test_switch_jnz_terminator(self):
        """
        2-block chain: jz + jnz → SwitchNode with case for jz, case for
        fall-through, and default_label from jnz.

        Topology:
            b0 (0x1000): A=R7; A+=0xFE(-2); jz label_c2 → [b_c2, b1]
            b1 (0x1010): A+=0xFC(-4);        jnz label_def → [b_def, b_fall]
            b_c2   (0x1020): label_c2
            b_def  (0x1030): label_def
            b_fall (0x1040): (fall-through, case_val = 6)

        After b0: cumulative=0xFE → case 2 → label_c2
        After b1: cumulative=(0xFE+0xFC)&FF=0xFA(-6) → case_val=6 → fall-through
        default → label_def
        """
        b_c2   = FakeBlock(0x1020, hir=[Statement(0x1020, "return;")], label="label_c2")
        b_def  = FakeBlock(0x1030, hir=[Statement(0x1030, "return;")], label="label_def")
        b_fall = FakeBlock(0x1040, hir=[Statement(0x1040, "return;")], label="label_fall")
        b1     = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFC)),
            IfGoto(0x1012, BinOp(Reg("A"), "!=", Const(0)), "label_def"),
        ])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2)
        connect(b0, b1)
        connect(b1, b_def)
        connect(b1, b_fall)

        func = FakeFunction("sw_jnz", [b0, b1, b_c2, b_def, b_fall])
        SwitchStructurer().run(func)

        assert len(b0.hir) == 1
        sw = b0.hir[0]
        assert isinstance(sw, SwitchNode)
        assert sw.subject == Reg("R7")
        assert sw.default_label == "label_def"
        all_cases = {v: lbl for vals, lbl in sw.cases for v in vals}
        assert all_cases[2] == "label_c2"
        assert all_cases[6] == "label_fall"   # fall-through case
        assert b1._absorbed

    def test_single_step_not_converted(self):
        """
        A single (jz) step does NOT become a SwitchNode — chain must be ≥ 2.
        """
        b_c2   = FakeBlock(0x1020, hir=[Statement(0x1020, "return;")], label="label_c2")
        b_fall = FakeBlock(0x1030, hir=[Statement(0x1030, "return;")])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2)
        connect(b0, b_fall)

        func = FakeFunction("sw_single", [b0, b_c2, b_fall])
        SwitchStructurer().run(func)

        # HIR should be unchanged (IfGoto still present)
        assert any(isinstance(n, IfGoto) for n in b0.hir)
        assert not isinstance(b0.hir[-1], SwitchNode)

    def test_duplicate_labels_merged(self):
        """
        When two steps jump to the same label, they are merged into one case entry.

        b0: A=R7; A+=0xFE(-2); jz label_x → [b_x, b1]
        b1: A+=0xFE(-2);        jz label_x → [b_x, b_fall]

        Both case 2 and case 4 go to label_x → merged into one entry.
        """
        b_x    = FakeBlock(0x1020, hir=[Statement(0x1020, "return;")], label="label_x")
        b_fall = FakeBlock(0x1030, hir=[Statement(0x1030, "return;")])
        b1     = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1012, BinOp(Reg("A"), "==", Const(0)), "label_x"),
        ])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_x"),
        ])

        connect(b0, b_x)
        connect(b0, b1)
        connect(b1, b_x)
        connect(b1, b_fall)

        func = FakeFunction("sw_merge", [b0, b1, b_x, b_fall])
        SwitchStructurer().run(func)

        sw = b0.hir[-1]
        assert isinstance(sw, SwitchNode)
        # Only one case entry (label_x) with both values
        assert len(sw.cases) == 1
        vals, lbl = sw.cases[0]
        assert lbl == "label_x"
        assert set(vals) == {2, 4}

    def test_five_block_chain_from_plan(self):
        """
        Full 5-block example from the plan (the user's assembly snippet).

        b0: A=R7; A+=0xFE; jz L2  → case 2
        b1: A+=0xFE;        jz L4  → case 4
        b2: A+=0xF4(-12);   jz L16 → case 16
        b3: A+=0xF0(-16);   jz L32 → case 32
        b4: A+=0x18(+24);   jnz Ld → default Ld; case 8 → fall-through
        """
        b_L2   = FakeBlock(0x2000, hir=[Statement(0x2000, "f2();")],  label="label_c2")
        b_L4   = FakeBlock(0x2010, hir=[Statement(0x2010, "f4();")],  label="label_c4")
        b_L16  = FakeBlock(0x2020, hir=[Statement(0x2020, "f16();")], label="label_c16")
        b_L32  = FakeBlock(0x2030, hir=[Statement(0x2030, "f32();")], label="label_c32")
        b_def  = FakeBlock(0x2040, hir=[Statement(0x2040, "fdef();")], label="label_def")
        b_fall = FakeBlock(0x2050, hir=[Statement(0x2050, "f8();")],  label="label_fall")

        b4 = FakeBlock(0x1040, hir=[
            CompoundAssign(0x1040, Reg("A"), "+=", Const(0x18)),
            IfGoto(0x1042, BinOp(Reg("A"), "!=", Const(0)), "label_def"),
        ])
        b3 = FakeBlock(0x1030, hir=[
            CompoundAssign(0x1030, Reg("A"), "+=", Const(0xF0)),
            IfGoto(0x1032, BinOp(Reg("A"), "==", Const(0)), "label_c32"),
        ])
        b2 = FakeBlock(0x1020, hir=[
            CompoundAssign(0x1020, Reg("A"), "+=", Const(0xF4)),
            IfGoto(0x1022, BinOp(Reg("A"), "==", Const(0)), "label_c16"),
        ])
        b1 = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1012, BinOp(Reg("A"), "==", Const(0)), "label_c4"),
        ])
        b0 = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_L2);  connect(b0, b1)
        connect(b1, b_L4);  connect(b1, b2)
        connect(b2, b_L16); connect(b2, b3)
        connect(b3, b_L32); connect(b3, b4)
        connect(b4, b_def); connect(b4, b_fall)

        func = FakeFunction("sw_5", [b0, b1, b2, b3, b4,
                                      b_L2, b_L4, b_L16, b_L32, b_def, b_fall])
        SwitchStructurer().run(func)

        sw = b0.hir[-1]
        assert isinstance(sw, SwitchNode)
        assert sw.subject == Reg("R7")
        assert sw.default_label == "label_def"

        all_cases = {v: lbl for vals, lbl in sw.cases for v in vals}
        assert all_cases[2]  == "label_c2"
        assert all_cases[4]  == "label_c4"
        assert all_cases[16] == "label_c16"
        assert all_cases[32] == "label_c32"
        assert all_cases[8]  == "label_fall"

        # Intermediate blocks absorbed; case targets and fall-through preserved
        assert b1._absorbed
        assert b2._absorbed
        assert b3._absorbed
        assert b4._absorbed
        assert not b_L2._absorbed
        assert not b_L4._absorbed
        assert not b_L16._absorbed
        assert not b_L32._absorbed
        assert not b_def._absorbed
        assert not b_fall._absorbed

    def test_switch_render(self):
        """SwitchNode.render() produces correct switch statement text."""
        sw = SwitchNode(
            ea=0x1000,
            subject=Reg("R7"),
            cases=[([2, 4], "label_c2"), ([16], "label_c16")],
            default_label="label_def",
        )
        lines = sw.render(indent=0)
        texts = [t for _, t in lines]
        assert texts[0] == "switch (R7) {"
        assert any("case 2: case 4: goto label_c2;" in t for t in texts)
        assert any("case 16: goto label_c16;" in t for t in texts)
        assert any("default: goto label_def;" in t for t in texts)
        assert texts[-1] == "}"
