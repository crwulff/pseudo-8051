"""
Tests for CJNEChainToSwitch: nested-if CJNE chain → SwitchNode rewriting.
"""

import pytest

from pseudo8051.ir.hir  import IfNode, SwitchNode, Assign
from pseudo8051.ir.hir.break_stmt import BreakStmt
from pseudo8051.ir.expr import Reg, Const, BinOp, UnaryOp
from pseudo8051.passes.cjne_switch import _detect_chain, _transform_nodes, _MIN_CASES


def _body_without_break(sw_body):
    """Strip the trailing BreakStmt added by _transform_nodes for comparison."""
    if sw_body and isinstance(sw_body[-1], BreakStmt):
        return sw_body[:-1]
    return sw_body


def _ne(reg, val):
    """BinOp: reg != val"""
    return BinOp(Reg(reg), '!=', Const(val))


def _eq(reg, val):
    """BinOp: reg == val"""
    return BinOp(Reg(reg), '==', Const(val))


def _assign(reg, val):
    return Assign(0, Reg(reg), Const(val))


def _cjne_chain(reg, cases, default_body=None, with_preamble=False):
    """
    Build a nested IfNode chain.
    cases: list of (val, body_nodes) in outermost-first order.
    default_body: then_nodes of innermost node (or None).
    with_preamble: if True, insert Assign(reg, Reg('R7')) before each inner IfNode.
    """
    if not cases:
        raise ValueError("need at least one case")

    val, body = cases[-1]
    inner = IfNode(0, _ne(reg, val), list(default_body) if default_body else [], list(body))

    for val, body in reversed(cases[:-1]):
        then_nodes = ([Assign(0, Reg(reg), Reg("R7"))] if with_preamble else []) + [inner]
        inner = IfNode(0, _ne(reg, val), then_nodes, list(body))
    return inner


class TestDetectChain:

    def test_simple_chain(self):
        body0 = [_assign("R6", 0)]
        body1 = [_assign("R6", 1)]
        body2 = [_assign("R6", 2)]
        node = _cjne_chain("R7", [(0, body0), (1, body1), (2, body2)])
        result = _detect_chain(node)
        assert result is not None
        reg, cases, default = result
        assert reg.name == "R7"
        assert [(v, b) for v, b in cases] == [(0, body0), (1, body1), (2, body2)]
        assert default is None

    def test_chain_with_default(self):
        body0 = [_assign("R6", 0)]
        body1 = [_assign("R6", 1)]
        default = [_assign("R6", 99)]
        node = _cjne_chain("R7", [(0, body0), (1, body1)], default_body=default)
        result = _detect_chain(node)
        assert result is not None
        _, cases, def_body = result
        assert len(cases) == 2
        assert def_body == default

    def test_equality_terminator(self):
        """Last case uses == instead of != (no else)."""
        body0 = [_assign("R6", 0)]
        body1 = [_assign("R6", 1)]
        body2 = [_assign("R6", 2)]
        # Build: if R7 != 0 { if R7 != 1 { if R7 == 2 { body2 } } else body1 } else body0
        eq_node  = IfNode(0, _eq("R7", 2), list(body2), [])
        mid      = IfNode(0, _ne("R7", 1), [eq_node], list(body1))
        outer    = IfNode(0, _ne("R7", 0), [mid], list(body0))
        result = _detect_chain(outer)
        assert result is not None
        _, cases, default = result
        assert len(cases) == 3
        assert cases[0] == (0, body0)
        assert cases[1] == (1, body1)
        assert cases[2] == (2, body2)
        assert default is None

    def test_not_a_chain_wrong_op(self):
        node = IfNode(0, BinOp(Reg("R7"), '<', Const(5)), [], [])
        assert _detect_chain(node) is None

    def test_not_a_chain_different_reg(self):
        """Chain breaks when a different register is compared."""
        body0 = [_assign("R6", 0)]
        body1 = [_assign("R6", 1)]
        inner = IfNode(0, _ne("R6", 1), [], list(body1))   # different reg!
        outer = IfNode(0, _ne("R7", 0), [inner], list(body0))
        result = _detect_chain(outer)
        # Only one != on R7 collected, inner uses R6 → default_body = [inner]
        assert result is not None
        _, cases, default = result
        assert len(cases) == 1   # only case 0 collected
        assert default == [inner]

    def test_reload_preamble_allowed(self):
        """then_nodes = [Assign(reg, reload), IfNode(reg != N)] is a valid chain link."""
        body0 = [_assign("R6", 0)]
        body1 = [_assign("R6", 1)]
        body2 = [_assign("R6", 2)]
        chain = _cjne_chain("A", [(0, body0), (1, body1), (2, body2)], with_preamble=True)
        result = _detect_chain(chain)
        assert result is not None
        _, cases, default = result
        assert len(cases) == 3
        assert default is None

    def test_not_ne_equality_terminator(self):
        """!(reg != val) with no else is treated as a final equality case."""
        body0 = [_assign("R6", 0)]
        body1 = [_assign("R6", 1)]
        body2 = [_assign("R6", 2)]
        eq_node = IfNode(0, UnaryOp('!', _ne("A", 2)), list(body2), [])
        mid     = IfNode(0, _ne("A", 1), [eq_node], list(body1))
        outer   = IfNode(0, _ne("A", 0), [mid], list(body0))
        result = _detect_chain(outer)
        assert result is not None
        _, cases, default = result
        assert len(cases) == 3
        assert cases[2] == (2, body2)
        assert default is None

    def test_not_an_if_node(self):
        node = _assign("R6", 42)
        assert _detect_chain(node) is None


class TestTransformNodes:

    def test_converts_chain_of_min_cases(self):
        body0 = [_assign("R6", 0)]
        body1 = [_assign("R6", 1)]
        body2 = [_assign("R6", 2)]
        chain = _cjne_chain("R7", [(0, body0), (1, body1), (2, body2)])
        result = _transform_nodes([chain])
        assert len(result) == 1
        assert isinstance(result[0], SwitchNode)
        sw = result[0]
        assert sw.subject == Reg("R7")
        assert len(sw.cases) == 3
        vals_to_bodies = {vals[0]: _body_without_break(body) for vals, body in sw.cases}
        assert vals_to_bodies[0] == body0
        assert vals_to_bodies[1] == body1
        assert vals_to_bodies[2] == body2
        assert sw.default_body is None
        # Every case body must end with a BreakStmt
        for _, body in sw.cases:
            assert isinstance(body[-1], BreakStmt)

    def test_below_min_cases_not_converted(self):
        """Chains shorter than MIN_CASES are left as IfNodes."""
        bodies = [[_assign("R6", i)] for i in range(_MIN_CASES - 1)]
        chain = _cjne_chain("R7", list(enumerate(bodies)))
        result = _transform_nodes([chain])
        assert isinstance(result[0], IfNode)

    def test_at_min_cases_converted(self):
        bodies = [[_assign("R6", i)] for i in range(_MIN_CASES)]
        chain = _cjne_chain("R7", list(enumerate(bodies)))
        result = _transform_nodes([chain])
        assert isinstance(result[0], SwitchNode)

    def test_non_consecutive_values(self):
        body0 = [_assign("R6", 0)]
        body5 = [_assign("R6", 5)]
        bodyA = [_assign("R6", 10)]
        chain = _cjne_chain("R7", [(0, body0), (5, body5), (10, bodyA)])
        result = _transform_nodes([chain])
        assert isinstance(result[0], SwitchNode)
        sw = result[0]
        vals_to_bodies = {vals[0]: _body_without_break(body) for vals, body in sw.cases}
        assert vals_to_bodies[0] == body0
        assert vals_to_bodies[5] == body5
        assert vals_to_bodies[10] == bodyA

    def test_with_default_body(self):
        body0 = [_assign("R6", 0)]
        body1 = [_assign("R6", 1)]
        body2 = [_assign("R6", 2)]
        default = [_assign("R6", 99)]
        chain = _cjne_chain("R7", [(0, body0), (1, body1), (2, body2)], default_body=default)
        result = _transform_nodes([chain])
        assert isinstance(result[0], SwitchNode)
        assert result[0].default_body == default

    def test_preserves_non_switch_nodes(self):
        node = _assign("R6", 42)
        result = _transform_nodes([node])
        assert result == [node]
