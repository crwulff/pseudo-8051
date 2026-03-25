from pseudo8051.ir.hir import Statement, IfNode
from pseudo8051.passes.ifelse import IfElseStructurer

from ..helpers import FakeBlock, FakeFunction, connect


class TestIfElseStructurer:

    def test_if_then_no_else(self):
        """
        Block A → [B (true), C (false/merge)], B → C.
        Result: A.hir = [IfNode(cond, then=[...], else=[])]
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
        assert "A != 0" in node.condition
        assert node.then_nodes[0].text == "R7 = 0;"
        assert node.else_nodes == []

    def test_dead_end_false_arm(self):
        """
        False arm is a terminal block (no successors) — the 'if-with-exit' pattern.
        """
        A = FakeBlock(0x1000, hir=[Statement(0x1000, "if (A != 0) goto label_b;")])
        C = FakeBlock(0x1010, hir=[Statement(0x1010, "return;")])
        B = FakeBlock(0x1020, hir=[Statement(0x1020, "R7 = 1;")], label="label_b")

        connect(A, B)   # true arm
        connect(A, C)   # false arm (dead-end)

        func = FakeFunction("dead_end_f", [A, C, B])
        IfElseStructurer().run(func)

        node = A.hir[0]
        assert isinstance(node, IfNode)
        assert "A != 0" in (node.condition if isinstance(node.condition, str)
                            else node.condition.render())
        assert len(node.then_nodes) == 1
        assert node.then_nodes[0].text == "return;"
        assert node.else_nodes == []
        assert C._absorbed

    def test_dead_end_false_arm_multiblock(self):
        """
        False arm spans two blocks before terminating — multi-block dead-end path.
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
        When the merge block has an IDA-assigned label, the goto must still be stripped.
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
        assert len(node.else_nodes) == 1
        assert node.else_nodes[0].text == "R7 = 1;"

    def test_dead_end_arm_skipped_if_externally_referenced(self):
        """
        E's dead-end false arm (F, label 'label_f') is also the target of A's goto.
        E must NOT be structured (F cannot be absorbed), but A can be structured.
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

        assert not F._absorbed
        assert B._absorbed
        assert E._absorbed
