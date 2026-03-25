from pseudo8051.ir.hir import Statement, ForNode, WhileNode
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
