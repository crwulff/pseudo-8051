from pseudo8051.ir.hir import Statement, ForNode, WhileNode, IfNode, IfGoto, GotoStatement
from pseudo8051.ir.expr import BinOp, Const, Reg
from pseudo8051.passes.loops import LoopStructurer

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

        assert len(header.hir) == 1
        fn = header.hir[0]
        assert isinstance(fn, ForNode)
        assert fn.init == "R7 = 5"
        assert _cond_str(fn.condition) == "R7"
        assert _cond_str(fn.update)    == "--R7"
        assert len(fn.body_nodes) == 1
        assert isinstance(fn.body_nodes[0], Statement)
        assert fn.body_nodes[0].text == "XRAM[S] = A;"

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

    def test_multi_tail_loop_produces_single_while_with_if(self):
        """
        Two back-edges to the same header:
          body (0x1010): R7=1; IfGoto(R7!=0 → header)  — conditional early-continue
          tail (0x1020): R6=1; GotoStatement(→ header)  — normal loop end

        Expected: one WhileNode whose body is [Statement("R7 = 1;"),
                                               IfNode(R7==0, ["R6 = 1;"])]
        Topology:
            header(0x1000): if(cond) goto exit  → [exit(0x2000), body(0x1010)]
            body  (0x1010): R7=1; jnz→header   → [header(back-edge), tail(0x1020)]
            tail  (0x1020): R6=1; sjmp→header  → [header(back-edge)]
            exit  (0x2000): return;
        """
        exit_b = FakeBlock(0x2000, hir=[Statement(0x2000, "return;")], label="label_exit")
        tail   = FakeBlock(0x1020, hir=[
            Statement(0x1020, "R6 = 1;"),
            GotoStatement(0x1022, "label_1000"),
        ])
        body   = FakeBlock(0x1010, hir=[
            Statement(0x1010, "R7 = 1;"),
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

        # body_nodes: [Statement("R7 = 1;"), IfNode(R7==0, ["R6 = 1;"])]
        body_texts = [n.text for n in wn.body_nodes if isinstance(n, Statement)]
        assert "R7 = 1;" in body_texts

        if_node = wn.body_nodes[-1]
        assert isinstance(if_node, IfNode)
        assert len(if_node.then_nodes) == 1
        assert if_node.then_nodes[0].text == "R6 = 1;"
        assert if_node.else_nodes == []

        # Condition is inverted: R7 != 0 → R7 == 0
        cond = if_node.condition
        assert isinstance(cond, BinOp) and cond.op == "=="

        # Both inner blocks absorbed; exit block not
        assert body._absorbed
        assert tail._absorbed
        assert not exit_b._absorbed
