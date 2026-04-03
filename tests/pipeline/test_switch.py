import pytest

from pseudo8051.ir.hir import (IfGoto, Assign, CompoundAssign,
                                SwitchNode, GotoStatement, BreakStmt, Label, ReturnStmt,
                                ExprStmt)
from pseudo8051.ir.expr import Reg, Const, BinOp, Name, Call
from pseudo8051.passes.switch import SwitchStructurer, SwitchBodyAbsorber
from pseudo8051.passes.ifelse import IfElseStructurer
from pseudo8051.passes.typesimplify import TypeAwareSimplifier
from pseudo8051.prototypes import PROTOTYPES, FuncProto, Param

from ..helpers import FakeBlock, FakeFunction, connect


class TestSwitchStructurer:
    """Tests for the SwitchStructurer pass."""

    def test_switch_basic_jz_chain(self):
        """3-block jz chain → SwitchNode with 2 cases, no default."""
        b_c2   = FakeBlock(0x1020, hir=[ReturnStmt(0x1020)], label="label_c2")
        b_c4   = FakeBlock(0x1030, hir=[ReturnStmt(0x1030)], label="label_c4")
        b_fall = FakeBlock(0x1040, hir=[ReturnStmt(0x1040)])
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
        assert len(sw.cases) == 2
        all_cases = {v: lbl for vals, lbl in sw.cases for v in vals}
        assert all_cases[2] == "label_c2"
        assert all_cases[4] == "label_c4"
        assert b1._absorbed
        assert not b_c2._absorbed
        assert not b_c4._absorbed

    def test_switch_jnz_terminator(self):
        """2-block chain: jz + jnz → SwitchNode with case for jz, case for fall-through, and default."""
        b_c2   = FakeBlock(0x1020, hir=[ReturnStmt(0x1020)], label="label_c2")
        b_def  = FakeBlock(0x1030, hir=[ReturnStmt(0x1030)], label="label_def")
        b_fall = FakeBlock(0x1040, hir=[ReturnStmt(0x1040)], label="label_fall")
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
        assert all_cases[6] == "label_fall"
        assert b1._absorbed

    def test_single_step_not_converted(self):
        """A single (jz) step does NOT become a SwitchNode — chain must be ≥ 2."""
        b_c2   = FakeBlock(0x1020, hir=[ReturnStmt(0x1020)], label="label_c2")
        b_fall = FakeBlock(0x1030, hir=[ReturnStmt(0x1030)])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2)
        connect(b0, b_fall)

        func = FakeFunction("sw_single", [b0, b_c2, b_fall])
        SwitchStructurer().run(func)

        assert any(isinstance(n, IfGoto) for n in b0.hir)
        assert not isinstance(b0.hir[-1], SwitchNode)

    def test_duplicate_labels_merged(self):
        """When two steps jump to the same label, they are merged into one case entry."""
        b_x    = FakeBlock(0x1020, hir=[ReturnStmt(0x1020)], label="label_x")
        b_fall = FakeBlock(0x1030, hir=[ReturnStmt(0x1030)])
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
        assert len(sw.cases) == 1
        vals, lbl = sw.cases[0]
        assert lbl == "label_x"
        assert set(vals) == {2, 4}

    def test_five_block_chain_from_plan(self):
        """Full 5-block example: b0..b4 with jz/jnz steps."""
        b_L2   = FakeBlock(0x2000, hir=[ExprStmt(0x2000, Call("f2",  []))], label="label_c2")
        b_L4   = FakeBlock(0x2010, hir=[ExprStmt(0x2010, Call("f4",  []))], label="label_c4")
        b_L16  = FakeBlock(0x2020, hir=[ExprStmt(0x2020, Call("f16", []))], label="label_c16")
        b_L32  = FakeBlock(0x2030, hir=[ExprStmt(0x2030, Call("f32", []))], label="label_c32")
        b_def  = FakeBlock(0x2040, hir=[ExprStmt(0x2040, Call("fdef",[]))], label="label_def")
        b_fall = FakeBlock(0x2050, hir=[ExprStmt(0x2050, Call("f8",  []))], label="label_fall")

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


class TestSwitchBodyAbsorber:
    """Tests for SwitchBodyAbsorber — inline case bodies into SwitchNode."""

    def test_basic_absorption(self):
        """2-case switch + default: case bodies inlined, gotos replaced with break."""
        merge  = FakeBlock(0x2030, hir=[
            Label(0x2030, "label_merge"),
            ReturnStmt(0x2030),
        ], label="label_merge")
        b_c2   = FakeBlock(0x2000, hir=[
            Label(0x2000, "label_c2"),
            Assign(0x2000, Reg("R7"), Const(2)),
            GotoStatement(0x2002, "label_merge"),
        ], label="label_c2")
        b_fall = FakeBlock(0x2010, hir=[
            Label(0x2010, "label_fall"),
            Assign(0x2010, Reg("R7"), Const(4)),
            GotoStatement(0x2012, "label_merge"),
        ], label="label_fall")
        b_def  = FakeBlock(0x2020, hir=[
            Label(0x2020, "label_def"),
            Assign(0x2020, Reg("R7"), Const(99)),
            GotoStatement(0x2022, "label_merge"),
        ], label="label_def")
        b1     = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1012, BinOp(Reg("A"), "!=", Const(0)), "label_def"),
        ])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2);   connect(b0, b1)
        connect(b1, b_def);  connect(b1, b_fall)
        connect(b_c2, merge); connect(b_fall, merge); connect(b_def, merge)

        func = FakeFunction("sw_absorb", [b0, b1, b_c2, b_fall, b_def, merge])

        SwitchStructurer().run(func)
        IfElseStructurer().run(func)
        SwitchBodyAbsorber().run(func)

        # merge block is absorbed inline after the switch when not externally referenced
        assert len(b0.hir) == 2
        sw = b0.hir[0]
        assert isinstance(sw, SwitchNode)
        assert isinstance(b0.hir[1], ReturnStmt)

        for values, body in sw.cases:
            assert isinstance(body, list), \
                f"case {values} body should be inlined, got {body!r}"

        assert sw.default_label is None
        assert sw.default_body is not None
        assert isinstance(sw.default_body, list)

        assert b_c2._absorbed
        assert b_fall._absorbed
        assert b_def._absorbed
        assert merge._absorbed

    def test_render_inline_bodies(self):
        """SwitchNode with inlined bodies renders correctly."""
        merge  = FakeBlock(0x2030, hir=[
            Label(0x2030, "label_merge"),
            ReturnStmt(0x2030),
        ], label="label_merge")
        b_c2   = FakeBlock(0x2000, hir=[
            Label(0x2000, "label_c2"),
            Assign(0x2000, Reg("R7"), Const(2)),
            GotoStatement(0x2002, "label_merge"),
        ], label="label_c2")
        b_fall = FakeBlock(0x2010, hir=[
            Label(0x2010, "label_fall"),
            Assign(0x2010, Reg("R7"), Const(4)),
            GotoStatement(0x2012, "label_merge"),
        ], label="label_fall")
        b1     = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1012, BinOp(Reg("A"), "==", Const(0)), "label_fall"),
        ])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2);   connect(b0, b1)
        connect(b1, b_fall); connect(b1, merge)
        connect(b_c2, merge); connect(b_fall, merge)

        func = FakeFunction("sw_render", [b0, b1, b_c2, b_fall, merge])

        SwitchStructurer().run(func)
        IfElseStructurer().run(func)
        SwitchBodyAbsorber().run(func)

        sw = b0.hir[0]
        assert isinstance(sw, SwitchNode)

        lines = sw.render(indent=0)
        texts = [t for _, t in lines]

        assert texts[0] == "switch (R7) {"
        assert texts[-1] == "}"
        assert not any("goto label_c2" in t for t in texts)
        assert not any("goto label_fall" in t for t in texts)
        assert any("R7 = 2;" in t for t in texts)
        assert any("R7 = 4;" in t for t in texts)
        assert any("break;" in t for t in texts)

    def test_case_body_ends_with_return_no_break(self):
        """Case body ending with return; should NOT get a trailing break;."""
        merge  = FakeBlock(0x2030, hir=[
            Label(0x2030, "label_merge"),
            Assign(0x2030, Reg("R6"), Const(0)),
        ], label="label_merge")
        b_c2   = FakeBlock(0x2000, hir=[
            Label(0x2000, "label_c2"),
            ReturnStmt(0x2000),
        ], label="label_c2")
        b_fall = FakeBlock(0x2010, hir=[
            Label(0x2010, "label_fall"),
            Assign(0x2010, Reg("R7"), Const(4)),
            GotoStatement(0x2012, "label_merge"),
        ], label="label_fall")
        b1     = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1012, BinOp(Reg("A"), "==", Const(0)), "label_fall"),
        ])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2);   connect(b0, b1)
        connect(b1, b_fall); connect(b1, merge)
        connect(b_c2, merge); connect(b_fall, merge)

        func = FakeFunction("sw_return", [b0, b1, b_c2, b_fall, merge])

        SwitchStructurer().run(func)
        IfElseStructurer().run(func)
        SwitchBodyAbsorber().run(func)

        sw = b0.hir[0]
        assert isinstance(sw, SwitchNode)

        for values, body in sw.cases:
            if isinstance(body, list):
                has_return = any(isinstance(n, ReturnStmt) for n in body)
                if has_return:
                    assert isinstance(body[-1], ReturnStmt), \
                        f"Expected ReturnStmt as last node, got {body[-1]!r}"
                    break
        else:
            pytest.fail("No case body with return found")

    def test_all_arms_terminate(self):
        """Switch where every case arm ends with ret (no merge point) → bodies still inlined."""
        b_c2   = FakeBlock(0x2000, hir=[
            Label(0x2000, "label_c2"),
            Assign(0x2000, Reg("R7"), Const(2)),
            ReturnStmt(0x2002),
        ], label="label_c2")
        b_fall = FakeBlock(0x2010, hir=[
            Label(0x2010, "label_fall"),
            Assign(0x2010, Reg("R7"), Const(4)),
            ReturnStmt(0x2012),
        ], label="label_fall")
        b_def  = FakeBlock(0x2020, hir=[
            Label(0x2020, "label_def"),
            Assign(0x2020, Reg("R7"), Const(99)),
            ReturnStmt(0x2022),
        ], label="label_def")
        b1     = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1012, BinOp(Reg("A"), "!=", Const(0)), "label_def"),
        ])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2);  connect(b0, b1)
        connect(b1, b_def); connect(b1, b_fall)
        # No merge block; all arms terminate with ret

        func = FakeFunction("sw_terminate", [b0, b1, b_c2, b_fall, b_def])

        SwitchStructurer().run(func)
        IfElseStructurer().run(func)
        SwitchBodyAbsorber().run(func)

        sw = b0.hir[0]
        assert isinstance(sw, SwitchNode)

        for values, body in sw.cases:
            assert isinstance(body, list), \
                f"case {values} body should be inlined, got {body!r}"
            assert not any(isinstance(n, BreakStmt) for n in body), \
                "no break; expected (arm ends with return)"

        assert sw.default_label is None
        assert isinstance(sw.default_body, list)
        assert b_c2._absorbed
        assert b_fall._absorbed
        assert b_def._absorbed

    def test_mixed_arms_some_terminate_some_break(self):
        """Switch where one arm returns and two arms fall through to a merge → all bodies inlined.

        Layout:
          b0: switch head  (case 2 → label_c2, falls into b1)
          b1: switch step  (case 4 → label_fall, default → label_def)
          b_c2:   R7 = 2; return;          ← dead-end arm
          b_fall: R7 = 4; goto label_merge ← live arm
          b_def:  R7 = 99; goto label_merge ← live arm (default)
          merge:  return;                  ← shared merge point for live arms
        """
        merge  = FakeBlock(0x3000, hir=[
            Label(0x3000, "label_merge"),
            ReturnStmt(0x3000),
        ], label="label_merge")
        b_c2   = FakeBlock(0x2000, hir=[
            Label(0x2000, "label_c2"),
            Assign(0x2000, Reg("R7"), Const(2)),
            ReturnStmt(0x2002),
        ], label="label_c2")
        b_fall = FakeBlock(0x2010, hir=[
            Label(0x2010, "label_fall"),
            Assign(0x2010, Reg("R7"), Const(4)),
            GotoStatement(0x2012, "label_merge"),
        ], label="label_fall")
        b_def  = FakeBlock(0x2020, hir=[
            Label(0x2020, "label_def"),
            Assign(0x2020, Reg("R7"), Const(99)),
            GotoStatement(0x2022, "label_merge"),
        ], label="label_def")
        b1     = FakeBlock(0x1010, hir=[
            CompoundAssign(0x1010, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1012, BinOp(Reg("A"), "!=", Const(0)), "label_def"),
        ])
        b0     = FakeBlock(0x1000, hir=[
            Assign(0x1000, Reg("A"), Reg("R7")),
            CompoundAssign(0x1002, Reg("A"), "+=", Const(0xFE)),
            IfGoto(0x1004, BinOp(Reg("A"), "==", Const(0)), "label_c2"),
        ])

        connect(b0, b_c2);  connect(b0, b1)
        connect(b1, b_def); connect(b1, b_fall)
        connect(b_fall, merge); connect(b_def, merge)

        func = FakeFunction("sw_mixed", [b0, b1, b_c2, b_fall, b_def, merge])

        SwitchStructurer().run(func)
        IfElseStructurer().run(func)
        SwitchBodyAbsorber().run(func)

        sw = b0.hir[0]
        assert isinstance(sw, SwitchNode)

        # All case bodies must be inlined (no dangling goto-label strings)
        for values, body in sw.cases:
            assert isinstance(body, list), \
                f"case {values} body should be inlined, got {body!r}"

        # default must be inlined too
        assert sw.default_label is None
        assert isinstance(sw.default_body, list)

        # The returning arm (case 2) must end with return, not break
        case_map = {v: body for vals, body in sw.cases for v in vals}
        assert not any(isinstance(n, BreakStmt) for n in case_map[2]), \
            "returning arm should not have break;"

        # The live arms (case 4 and default) must end with break (goto→merge replaced)
        assert any(isinstance(n, BreakStmt) for n in case_map[4]), \
            "live arm should end with break;"
        assert any(isinstance(n, BreakStmt) for n in sw.default_body), \
            "default live arm should end with break;"

        assert b_c2._absorbed
        assert b_fall._absorbed
        assert b_def._absorbed
        assert merge._absorbed  # merge block absorbed inline after switch when not externally referenced

    def test_identical_bodies_merged(self):
        """Cases that map to different labels but identical body HIR are merged.

        This happens when a JMP @A+DPTR table has multiple entries pointing at
        the same code block but with different labels assigned.

        Layout:
          switch head: case 0 → label_a, case 1 → label_b, case 2 → label_c
          b_a (label_a): R7 = 1; goto merge   ← same body as b_b
          b_b (label_b): R7 = 1; goto merge   ← same body as b_a
          b_c (label_c): R7 = 99; goto merge  ← distinct body
          merge: return;

        After absorption + dedup: cases 0 and 1 are merged into one entry.
        """
        merge = FakeBlock(0x3000, hir=[
            Label(0x3000, "label_merge"),
            ReturnStmt(0x3000),
        ], label="label_merge")
        b_c = FakeBlock(0x2020, hir=[
            Label(0x2020, "label_c"),
            Assign(0x2020, Reg("R7"), Const(99)),
            GotoStatement(0x2022, "label_merge"),
        ], label="label_c")
        b_b = FakeBlock(0x2010, hir=[
            Label(0x2010, "label_b"),
            Assign(0x2010, Reg("R7"), Const(1)),
            GotoStatement(0x2012, "label_merge"),
        ], label="label_b")
        b_a = FakeBlock(0x2000, hir=[
            Label(0x2000, "label_a"),
            Assign(0x2000, Reg("R7"), Const(1)),
            GotoStatement(0x2002, "label_merge"),
        ], label="label_a")
        # Pre-built SwitchNode (as JmpTableStructurer would produce)
        sw = SwitchNode(
            ea=0x1000,
            subject=Reg("R7"),
            cases=[([0], "label_a"), ([1], "label_b"), ([2], "label_c")],
            default_label=None,
        )
        head = FakeBlock(0x1000, hir=[sw])

        connect(head, b_a); connect(head, b_b); connect(head, b_c)
        connect(b_a, merge); connect(b_b, merge); connect(b_c, merge)

        func = FakeFunction("sw_dedup", [head, b_a, b_b, b_c, merge])
        SwitchBodyAbsorber().run(func)

        sw_out = head.hir[0]
        assert isinstance(sw_out, SwitchNode)

        # Should have exactly 2 cases after merging 0 and 1
        assert len(sw_out.cases) == 2, \
            f"expected 2 cases after dedup, got {len(sw_out.cases)}: {sw_out.cases}"
        case_map = {v: body for vals, body in sw_out.cases for v in vals}
        assert set(case_map.keys()) == {0, 1, 2}
        # Cases 0 and 1 share the same body list
        assert case_map[0] is case_map[1], "cases 0 and 1 should share the same body"


class TestSwitchNodeTypeSimplify:
    """TypeAwareSimplifier should substitute the SwitchNode subject and simplify case bodies."""

    def setup_method(self):
        PROTOTYPES.clear()

    def teardown_method(self):
        PROTOTYPES.clear()

    def test_subject_substituted_with_proto(self):
        """SwitchNode(R7, ...) with proto arg1→R7 → subject becomes Name("arg1")."""
        PROTOTYPES["sw_proto"] = FuncProto(
            return_type="void",
            params=[Param("arg1", "uint8_t", ("R7",))],
        )
        sw = SwitchNode(
            ea=0x1000,
            subject=Reg("R7"),
            cases=[([2], "label_c2"), ([4], "label_c4")],
            default_label="label_def",
        )
        block = FakeBlock(0x1000, hir=[sw], live_in=frozenset({"R7"}))
        func = FakeFunction("sw_proto", [block], hir=[sw])

        TypeAwareSimplifier().run(func)

        assert len(func.hir) == 1
        sw_out = func.hir[0]
        assert isinstance(sw_out, SwitchNode)
        assert sw_out.subject == Name("arg1"), \
            f"Expected Name('arg1'), got {sw_out.subject!r}"

    def test_subject_substituted_from_liveness(self):
        """Without proto, SwitchNode(R7, ...) with R7 live-in → subject becomes Name('arg1')."""
        sw = SwitchNode(
            ea=0x1000,
            subject=Reg("R7"),
            cases=[([1], "label_c1"), ([3], "label_c3")],
            default_label=None,
        )
        block = FakeBlock(0x1000, hir=[sw], live_in=frozenset({"R7"}))
        func = FakeFunction("sw_noproto", [block], hir=[sw])

        TypeAwareSimplifier().run(func)

        assert len(func.hir) == 1
        sw_out = func.hir[0]
        assert isinstance(sw_out, SwitchNode)
        assert sw_out.subject == Name("arg1"), \
            f"Expected Name('arg1'), got {sw_out.subject!r}"

    def test_case_body_simplified(self):
        """Case body containing A = R7 relay gets simplified when R7 live-in."""
        case_body = [
            Label(0x2000, "label_c2"),
            Assign(0x2002, Reg("A"), Reg("R7")),
            Assign(0x2004, Reg("R6"), Reg("A")),
            BreakStmt(0x2006),
        ]
        sw = SwitchNode(
            ea=0x1000,
            subject=Reg("R7"),
            cases=[([2], case_body)],
            default_label=None,
        )
        block = FakeBlock(0x1000, hir=[sw], live_in=frozenset({"R7"}))
        func = FakeFunction("sw_body_simp", [block], hir=[sw])

        TypeAwareSimplifier().run(func)

        sw_out = func.hir[0]
        assert isinstance(sw_out, SwitchNode)
        _vals, body = sw_out.cases[0]
        # A = R7 relay should be absorbed (R6 = R7 or R6 = arg1, not two separate nodes)
        assigns = [n for n in body if isinstance(n, Assign)]
        a_assigns = [n for n in assigns if n.lhs == Reg("A")]
        assert not a_assigns, "A = R7 relay should be collapsed"
