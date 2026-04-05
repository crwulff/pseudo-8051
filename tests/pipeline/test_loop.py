from pseudo8051.ir.hir import Assign, ExprStmt, ReturnStmt, ForNode, WhileNode, DoWhileNode, IfNode, IfGoto, GotoStatement, CompoundAssign
from pseudo8051.ir.expr import BinOp, Const, Reg, Regs, UnaryOp, Name, XRAMRef
from pseudo8051.passes.loops import LoopStructurer, _dfs_back_edges

from ..helpers import FakeBlock, FakeFunction, connect


def _cond_str(c) -> str:
    return c.render() if hasattr(c, "render") else c


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
        init   = FakeBlock(0x1000, hir=[Assign(0x1000, Reg("R7"), Const(5))])
        header = FakeBlock(0x1010, hir=[])
        body   = FakeBlock(0x1018, hir=[
            Assign(0x1018, XRAMRef(Name("S")), Reg("A")),
            IfGoto(0x101a, BinOp(UnaryOp("--", Reg("R7")), "!=", Const(0)), "label_1010"),
        ])
        after  = FakeBlock(0x1020, hir=[])

        connect(init,   header)
        connect(header, body)
        connect(body,   header)   # back-edge
        connect(body,   after)

        func = FakeFunction("djnz_f", [init, header, body, after])
        LoopStructurer().run(func)

        assert len(header.hir) == 1
        fn = header.hir[0]
        assert isinstance(fn, ForNode)
        assert fn.init == "R7 = 5"
        assert _cond_str(fn.condition) == "R7"
        assert _cond_str(fn.update)    == "--R7"
        assert len(fn.body_nodes) == 1
        assert isinstance(fn.body_nodes[0], Assign)
        assert fn.body_nodes[0].render()[0][1] == "XRAM[S] = A;"

        assert body._absorbed

    def test_djnz_no_init_produces_while(self):
        """DJNZ loop without a detectable init value → WhileNode."""
        header = FakeBlock(0x1000, hir=[])
        body   = FakeBlock(0x1008, hir=[
            IfGoto(0x1008, BinOp(UnaryOp("--", Reg("R7")), "!=", Const(0)), "label_1000"),
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
            Assign(0x1000, XRAMRef(Name("S")), Reg("A")),
            IfGoto(0x1002, BinOp(Reg("C"), "!=", Const(0)), "label_1000"),
        ])
        after  = FakeBlock(0x1010, hir=[])

        connect(header, header)   # self-loop (back-edge)
        connect(header, after)

        func = FakeFunction("while_f", [header, after])
        LoopStructurer().run(func)

        assert len(header.hir) == 1
        wn = header.hir[0]
        assert isinstance(wn, WhileNode)
        assert _cond_str(wn.condition) == "C != 0"
        assert len(wn.body_nodes) == 1
        assert wn.body_nodes[0].render()[0][1] == "XRAM[S] = A;"

    def test_no_loop_no_change(self):
        """Linear CFG without back-edges is left unchanged."""
        b0 = FakeBlock(0x1000, hir=[Assign(0x1000, Reg("A"), Reg("R7"))])
        b1 = FakeBlock(0x1010, hir=[ReturnStmt(0x1010)])
        connect(b0, b1)
        func = FakeFunction("linear", [b0, b1])
        LoopStructurer().run(func)
        assert b0.hir[0].render()[0][1] == "A = R7;"
        assert b1.hir[0].render()[0][1] == "return;"

    def test_multi_tail_loop_produces_single_while_with_if(self):
        """
        Two back-edges to the same header:
          body (0x1010): R7=1; IfGoto(R7!=0 → header)  — conditional early-continue
          tail (0x1020): R6=1; GotoStatement(→ header)  — normal loop end

        Expected: one WhileNode whose body is [Assign(R7, 1),
                                               IfNode(R7==0, [Assign(R6, 1)])]
        Topology:
            header(0x1000): if(cond) goto exit  → [exit(0x2000), body(0x1010)]
            body  (0x1010): R7=1; jnz→header   → [header(back-edge), tail(0x1020)]
            tail  (0x1020): R6=1; sjmp→header  → [header(back-edge)]
            exit  (0x2000): return;
        """
        exit_b = FakeBlock(0x2000, hir=[ReturnStmt(0x2000)], label="label_exit")
        tail   = FakeBlock(0x1020, hir=[
            Assign(0x1020, Reg("R6"), Const(1)),
            GotoStatement(0x1022, "label_1000"),
        ])
        body   = FakeBlock(0x1010, hir=[
            Assign(0x1010, Reg("R7"), Const(1)),
            IfGoto(0x1012, BinOp(Reg("R7"), "!=", Const(0)), "label_1000"),
        ])
        header = FakeBlock(0x1000, hir=[
            IfGoto(0x1000, BinOp(Reg("A"), "!=", Const(4)), "label_exit"),
        ], label="label_1000")

        connect(header, exit_b)   # if-taken → exit
        connect(header, body)     # fall-through → loop body
        connect(body,   header)   # back-edge (conditional jnz)
        connect(body,   tail)     # fall-through to tail
        connect(tail,   header)   # back-edge (unconditional sjmp)

        func = FakeFunction("multi_tail_f", [header, body, tail, exit_b])
        LoopStructurer().run(func)

        # Exactly one WhileNode in header (plus a Label node for label_1000)
        while_nodes = [n for n in header.hir if isinstance(n, WhileNode)]
        assert len(while_nodes) == 1
        wn = while_nodes[0]

        # body_nodes: [Assign(R7, 1), IfNode(R7==0, [Assign(R6, 1)])]
        body_texts = [t for n in wn.body_nodes for _, t in n.render()]
        assert "R7 = 1;" in body_texts

        if_node = wn.body_nodes[-1]
        assert isinstance(if_node, IfNode)
        assert len(if_node.then_nodes) == 1
        assert if_node.then_nodes[0].render()[0][1] == "R6 = 1;"
        assert if_node.else_nodes == []

        # Condition is inverted: R7 != 0 → R7 == 0
        cond = if_node.condition
        assert isinstance(cond, BinOp) and cond.op == "=="

        # Both inner blocks absorbed; exit block not
        assert body._absorbed
        assert tail._absorbed
        assert not exit_b._absorbed

    def test_dfs_detects_high_address_header(self):
        """
        DFS correctly detects a loop whose header has a higher EA than the tail.
        With address-based comparison (succ.ea <= block.ea) this would be missed.

        Topology (entry=0x1000):
            init   (0x1000): → header (0x3000)
            body   (0x2000): GotoStatement → label_3000   [back-edge to header]
            header (0x3000): IfGoto(A == 0, label_exit)   → [exit (0x4000), body (0x2000)]
            exit   (0x4000)
        """
        exit_b = FakeBlock(0x4000, hir=[ReturnStmt(0x4000)], label="label_exit")
        body   = FakeBlock(0x2000, hir=[
            ExprStmt(0x2000, UnaryOp("++", Reg("R7"), post=True)),
            GotoStatement(0x2002, "label_3000"),
        ])
        header = FakeBlock(0x3000, hir=[
            IfGoto(0x3000, BinOp(Reg("A"), "==", Const(0)), "label_exit"),
        ], label="label_3000")
        init   = FakeBlock(0x1000, hir=[Assign(0x1000, Reg("A"), Reg("R7"))])

        connect(init,   header)
        connect(header, exit_b)   # branch-taken → exit
        connect(header, body)     # fall-through → loop body
        connect(body,   header)   # back-edge (higher EA)

        # Verify DFS detects the back-edge
        func = FakeFunction("high_header_f", [init, header, body, exit_b])
        back = _dfs_back_edges(func.entry_block)
        assert len(back) == 1
        tail_b, hdr_b = back[0]
        assert hdr_b.start_ea == 0x3000
        assert tail_b.start_ea == 0x2000

        # Verify LoopStructurer structures it correctly
        LoopStructurer().run(func)
        while_nodes = [n for n in header.hir if isinstance(n, WhileNode)]
        assert len(while_nodes) == 1
        wn = while_nodes[0]
        # IfGoto(A == 0, exit) → inverted → A != 0
        assert isinstance(wn.condition, BinOp)
        assert wn.condition.op == "!="
        assert body._absorbed

    def test_do_while_loop(self):
        """
        Do-while pattern: header has no branch; primary tail ends with a
        conditional back-edge → DoWhileNode.

        Topology:
            header (0x1000): "R7++;"  → body
            body   (0x1010): "R6 = 1;"
                             IfGoto(C != 0 → label_1000)  [back-edge]
                             → [header (back-edge), after (0x1020)]
            after  (0x1020): (empty)
        """
        after  = FakeBlock(0x1020, hir=[])
        body   = FakeBlock(0x1010, hir=[
            Assign(0x1010, Reg("R6"), Const(1)),
            IfGoto(0x1012, BinOp(Reg("C"), "!=", Const(0)), "label_1000"),
        ])
        header = FakeBlock(0x1000, hir=[
            ExprStmt(0x1000, UnaryOp("++", Reg("R7"), post=True)),
        ], label="label_1000")

        connect(header, body)
        connect(body,   header)   # back-edge
        connect(body,   after)

        func = FakeFunction("dowhile_f", [header, body, after])
        LoopStructurer().run(func)

        dw_nodes = [n for n in header.hir if isinstance(n, DoWhileNode)]
        assert len(dw_nodes) == 1
        dw = dw_nodes[0]
        assert _cond_str(dw.condition) == "C != 0"
        # body_nodes = [ExprStmt(R7++), Assign(R6, 1)]
        assert len(dw.body_nodes) == 2
        assert dw.body_nodes[0].render()[0][1] == "R7++;"
        assert dw.body_nodes[1].render()[0][1] == "R6 = 1;"
        assert body._absorbed

    def test_do_while_exit_condition_inverted(self):
        """
        Do-while: tail IfGoto targets exit (outside body) → condition inverted.

        Topology:
            header (0x1000): "R7++;"  → body
            body   (0x1010): IfGoto(C == 0 → label_exit)  → [exit, header (back-edge)]
            exit   (0x2000)
        """
        exit_b = FakeBlock(0x2000, hir=[], label="label_exit")
        body   = FakeBlock(0x1010, hir=[
            IfGoto(0x1010, BinOp(Reg("C"), "==", Const(0)), "label_exit"),
        ])
        header = FakeBlock(0x1000, hir=[
            ExprStmt(0x1000, UnaryOp("++", Reg("R7"), post=True)),
        ], label="label_1000")

        connect(header, body)
        connect(body,   exit_b)   # branch-taken → exit
        connect(body,   header)   # fall-through = back-edge

        func = FakeFunction("dowhile_exit_f", [header, body, exit_b])
        LoopStructurer().run(func)

        dw_nodes = [n for n in header.hir if isinstance(n, DoWhileNode)]
        assert len(dw_nodes) == 1
        dw = dw_nodes[0]
        # C == 0 → exit means C != 0 keeps looping
        assert _cond_str(dw.condition) == "C != 0"

    def test_while_promoted_to_for(self):
        """
        WhileNode is promoted to ForNode when the last body statement updates
        the loop condition variable.

        Topology:
            header (0x1000): IfGoto(R7 == 0, exit_label)  → [exit, body]
            body   (0x1010): "A++;", CompoundAssign(R7, "-=", Const(1))
                             → header (back-edge)
            exit   (0x2000)
        """
        exit_b = FakeBlock(0x2000, hir=[], label="label_exit")
        body   = FakeBlock(0x1010, hir=[
            ExprStmt(0x1010, UnaryOp("++", Reg("A"), post=True)),
            CompoundAssign(0x1012, Reg("R7"), "-=", Const(1)),
        ])
        header = FakeBlock(0x1000, hir=[
            IfGoto(0x1000, BinOp(Reg("R7"), "==", Const(0)), "label_exit"),
        ], label="label_1000")

        connect(header, exit_b)   # branch-taken → exit
        connect(header, body)     # fall-through → loop body
        connect(body,   header)   # back-edge

        func = FakeFunction("for_promote_f", [header, body, exit_b])
        LoopStructurer().run(func)

        loop_nodes = [n for n in header.hir
                      if isinstance(n, (WhileNode, ForNode))]
        assert len(loop_nodes) == 1
        fn = loop_nodes[0]
        assert isinstance(fn, ForNode)
        assert fn.init is None
        # condition: R7 == 0 inverted → R7 != 0
        assert _cond_str(fn.condition) == "R7 != 0"
        assert _cond_str(fn.update) == "R7 -= 1"
        # body_nodes has only "A++;" (last update node was hoisted)
        assert len(fn.body_nodes) == 1
        assert fn.body_nodes[0].render()[0][1] == "A++;"
        assert body._absorbed

    def test_forward_exit_condition_inverted(self):
        """JNC-style forward exit: while condition should be inverted (C, not !C)."""
        exit_b = FakeBlock(0x2020, hir=[ReturnStmt(0x2020)], label="label_exit")
        body   = FakeBlock(0x1010, hir=[
            Assign(0x1010, Reg("A"), Reg("R5")),
            GotoStatement(0x1012, "label_1000"),
        ])
        header = FakeBlock(0x1000, hir=[
            IfGoto(0x1000, UnaryOp("!", Reg("C")), "label_exit"),
        ], label="label_1000")
        connect(header, exit_b)   # branch-taken → exit
        connect(header, body)     # fall-through → body
        connect(body,   header)   # back-edge

        func = FakeFunction("jnc_f", [header, body, exit_b])
        LoopStructurer().run(func)

        while_nodes = [n for n in header.hir if isinstance(n, WhileNode)]
        assert len(while_nodes) == 1
        wn = while_nodes[0]
        # !C inverted to C
        assert isinstance(wn.condition, Regs) and wn.condition.is_single and wn.condition.name == "C"

    def test_djnz_in_orphan_block(self):
        """
        Loop inside a switch case body (orphan block — no predecessors).

        LoopStructurer must also DFS from blocks that have no CFG predecessors
        (e.g. JMP @A+DPTR switch case targets that IDA doesn't wire up).

        Topology:
            entry (0x1000): → b_case (but NO recorded predecessor edge on b_case)
            b_case(0x2000): body; if (--R7 != 0) goto label_2000  [self-loop]
                            → [b_case (back-edge), after (0x2010)]
            after (0x2010): (empty)
        """
        after  = FakeBlock(0x2010, hir=[], label="label_after")
        b_case = FakeBlock(0x2000, hir=[
            Assign(0x2000, XRAMRef(Name("S")), Reg("A")),
            IfGoto(0x2002, BinOp(UnaryOp("--", Reg("R7")), "!=", Const(0)), "label_2000"),
        ], label="label_2000")
        entry  = FakeBlock(0x1000, hir=[GotoStatement(0x1000, "label_2000")])

        # entry → b_case exists logically, but b_case has no predecessor recorded
        # (simulating IDA's behaviour for JMP @A+DPTR targets)
        entry._succs.append(b_case)
        # b_case self-loop and fall-through to after
        b_case._succs.append(b_case)
        b_case._succs.append(after)
        b_case._preds.append(b_case)   # only self-loop pred; no entry pred
        after._preds.append(b_case)

        func = FakeFunction("orphan_loop", [entry, b_case, after])
        LoopStructurer().run(func)

        loop_nodes = [n for n in b_case.hir if isinstance(n, DoWhileNode)]
        assert len(loop_nodes) == 1
        dw = loop_nodes[0]
        assert _cond_str(dw.condition) == "--R7 != 0"
        assert len(dw.body_nodes) == 1
        assert dw.body_nodes[0].render()[0][1] == "XRAM[S] = A;"

    def test_multibyte_djnz_produces_dowhile(self):
        """
        Multi-byte DJNZ: header self-loops on --R7 and primary tail loops on --R6.
        Together they decrement R6R7, so the result should be DoWhileNode(--R6R7 != 0).

        Topology:
            header (0x1000): body_op; if (--R7 != 0) goto label_1000  [back-edge]
                             → [header (back-edge), tail (0x1010)]
            tail   (0x1010): if (--R6 != 0) goto label_1000            [back-edge]
                             → [header (back-edge), after (0x1020)]
            after  (0x1020): (empty)

        Expected: header.hir = [DoWhileNode(--R6R7 != 0, [body_op])]
        """
        after  = FakeBlock(0x1020, hir=[])
        tail   = FakeBlock(0x1010, hir=[
            IfGoto(0x1010, BinOp(UnaryOp("--", Reg("R6")), "!=", Const(0)), "label_1000"),
        ])
        header = FakeBlock(0x1000, hir=[
            Assign(0x1000, XRAMRef(Name("S")), Reg("A")),
            IfGoto(0x1002, BinOp(UnaryOp("--", Reg("R7")), "!=", Const(0)), "label_1000"),
        ], label="label_1000")

        connect(header, header)   # self-loop (lo-byte back-edge)
        connect(header, tail)
        connect(tail,   header)   # hi-byte back-edge
        connect(tail,   after)

        func = FakeFunction("multibyte_djnz", [header, tail, after])
        LoopStructurer().run(func)

        dw_nodes = [n for n in header.hir if isinstance(n, DoWhileNode)]
        assert len(dw_nodes) == 1, f"expected DoWhileNode, got: {header.hir}"
        dw = dw_nodes[0]
        assert _cond_str(dw.condition) == "--R6R7 != 0"
        assert len(dw.body_nodes) == 1
        assert dw.body_nodes[0].render()[0][1] == "XRAM[S] = A;"
        assert tail._absorbed
