from pseudo8051.passes.typesimplify._post import _prune_orphaned_dptr_inc
from pseudo8051.ir.hir import ExprStmt, Assign, WhileNode, IfNode
from pseudo8051.ir.expr import Reg, Name, Const, UnaryOp, BinOp, XRAMRef, Cast


def _dptr_inc(ea=0):
    return ExprStmt(ea, UnaryOp("++", Reg("DPTR")))


class TestPruneOrphanedDptrInc:

    def test_orphaned_pruned(self):
        """DPTR++ with no downstream DPTR reference → removed."""
        nodes = [
            _dptr_inc(0),
            Assign(1, Name("x"), Name("y")),
        ]
        result = _prune_orphaned_dptr_inc(nodes)
        assert len(result) == 1
        assert isinstance(result[0], Assign)

    def test_dptr_used_kept(self):
        """DPTR++ where DPTR is used after → kept."""
        nodes = [
            _dptr_inc(0),
            Assign(1, Name("x"), Reg("DPTR")),
        ]
        result = _prune_orphaned_dptr_inc(nodes)
        assert len(result) == 2

    def test_inner_orphan_in_while_pruned(self):
        """DPTR++ inside WhileNode body with no downstream DPTR → removed."""
        body = [
            _dptr_inc(0),
            Assign(1, Name("x"), Name("y")),
        ]
        nodes = [WhileNode(0, BinOp(Name("n"), "!=", Const(0)), body)]
        result = _prune_orphaned_dptr_inc(nodes)
        assert isinstance(result[0], WhileNode)
        assert len(result[0].body_nodes) == 1  # DPTR++ removed

    def test_dptr_overwrite_kills_outer(self):
        """DPTR++ followed by DPTR=sym (kill) then XRAM[DPTR] inside IfNode → outer pruned."""
        while_body = [
            Assign(10, XRAMRef(Reg("DPTR")), Name("val")),
        ]
        if_then = [
            Assign(5, Reg("DPTR"), Name("_dest")),   # kills outer DPTR++ value
            WhileNode(6, BinOp(Name("n"), "!=", Const(0)), while_body),
        ]
        nodes = [
            _dptr_inc(0),               # outer orphan — killed by DPTR=_dest in if_then
            Assign(1, Name("x"), Name("y")),
            IfNode(2, BinOp(Name("n"), "!=", Const(0)), if_then, []),
        ]
        result = _prune_orphaned_dptr_inc(nodes)
        # DPTR++ at top should be pruned (DPTR=_dest kills it before XRAM[DPTR] is reached)
        assert len(result) == 2
        assert isinstance(result[0], Assign)
        assert isinstance(result[1], IfNode)

    def test_outer_orphan_removed_when_inner_pruned(self):
        """Top-level DPTR++ kept only because of inner DPTR++; after inner is pruned, outer is too."""
        inner_body = [
            _dptr_inc(10),
            Assign(11, Name("x"), Const(1)),
        ]
        nodes = [
            _dptr_inc(0),
            Assign(1, Name("y"), Const(2)),
            WhileNode(2, BinOp(Name("n"), "!=", Const(0)), inner_body),
        ]
        result = _prune_orphaned_dptr_inc(nodes)
        # Both DPTR++ should be gone; WhileNode body reduced to just Assign
        assert len(result) == 2  # Assign("y=2") + WhileNode
        assert isinstance(result[0], Assign)
        while_node = result[1]
        assert isinstance(while_node, WhileNode)
        assert len(while_node.body_nodes) == 1  # inner DPTR++ pruned
