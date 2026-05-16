"""
Tests for ConditionNormalizer: post-structuring IfNode condition normalization.
"""

from pseudo8051.ir.hir         import Assign, HIRNode
from pseudo8051.ir.hir.if_node import IfNode
from pseudo8051.ir.expr        import Reg, Const, BinOp, UnaryOp
from pseudo8051.passes.ifelse  import _normalize_hir_list

from tests.helpers import FakeBlock, FakeFunction


def _if(cond, then_nodes, else_nodes=None):
    return IfNode(0x1000, cond, then_nodes, else_nodes or [])


def _body():
    return [Assign(0x1010, Reg("R0"), Const(1))]


# ── Rule 1: !cond with both arms → swap arms ──────────────────────────────────

def test_not_cond_with_else_swaps_arms():
    """if (!c) {A} else {B} → if (c) {B} else {A}"""
    c = BinOp(Reg("A"), "==", Const(0))
    then_b = [Assign(0x1010, Reg("R0"), Const(1))]
    else_b = [Assign(0x1010, Reg("R1"), Const(2))]
    nodes = [_if(UnaryOp("!", c), then_b, else_b)]
    result = _normalize_hir_list(nodes)
    node = result[0]
    assert isinstance(node, IfNode)
    assert node.condition == c
    assert node.then_nodes is else_b
    assert node.else_nodes is then_b


def test_not_cond_no_else_unchanged():
    """if (!c) {A} (no else) → unchanged (Rule 1 requires both arms)."""
    c = BinOp(Reg("A"), "==", Const(0))
    cond = UnaryOp("!", c)
    then_b = _body()
    nodes = [_if(cond, then_b)]
    result = _normalize_hir_list(nodes)
    node = result[0]
    assert isinstance(node, IfNode)
    assert node.condition == cond
    assert not node.else_nodes


def test_not_not_cond_unchanged():
    """if (!!c) — double negation is NOT simplified by this pass."""
    c = Reg("A")
    cond = UnaryOp("!", UnaryOp("!", c))
    then_b = _body()
    else_b = _body()
    nodes = [_if(cond, then_b, else_b)]
    result = _normalize_hir_list(nodes)
    # The outer ! is stripped (has else), leaving UnaryOp("!", c) as condition
    # and arms swapped — but NOT double-stripped
    node = result[0]
    assert isinstance(node.condition, UnaryOp)
    assert node.condition.op == "!"
    assert node.condition.operand == c


# ── Rule 2: > and >= flipped to < and <= ─────────────────────────────────────

def test_gt_flipped_to_lt():
    """if (a > b) → if (b < a)"""
    a, b = Reg("A"), Reg("R0")
    nodes = [_if(BinOp(a, ">", b), _body())]
    result = _normalize_hir_list(nodes)
    node = result[0]
    assert isinstance(node.condition, BinOp)
    assert node.condition.op == "<"
    assert node.condition.lhs == b
    assert node.condition.rhs == a


def test_ge_flipped_to_le():
    """if (a >= b) → if (b <= a)"""
    a, b = Reg("A"), Const(5)
    nodes = [_if(BinOp(a, ">=", b), _body())]
    result = _normalize_hir_list(nodes)
    node = result[0]
    assert isinstance(node.condition, BinOp)
    assert node.condition.op == "<="
    assert node.condition.lhs == b
    assert node.condition.rhs == a


def test_lt_unchanged():
    """if (a < b) — already canonical, untouched."""
    cond = BinOp(Reg("A"), "<", Const(5))
    nodes = [_if(cond, _body())]
    result = _normalize_hir_list(nodes)
    assert result[0].condition == cond


def test_le_unchanged():
    """if (a <= b) — already canonical, untouched."""
    cond = BinOp(Reg("A"), "<=", Const(5))
    nodes = [_if(cond, _body())]
    result = _normalize_hir_list(nodes)
    assert result[0].condition == cond


def test_eq_unchanged():
    """if (a == b) — not affected by either rule."""
    cond = BinOp(Reg("A"), "==", Const(0))
    nodes = [_if(cond, _body())]
    result = _normalize_hir_list(nodes)
    assert result[0].condition == cond


# ── Both rules applied in sequence ───────────────────────────────────────────

def test_not_gt_swaps_then_flips():
    """if (!(a > b)) {A} else {B} → if (b < a) {B} else {A}"""
    a, b = Reg("A"), Reg("R0")
    then_b = [Assign(0x1010, Reg("R0"), Const(1))]
    else_b = [Assign(0x1010, Reg("R1"), Const(2))]
    nodes = [_if(UnaryOp("!", BinOp(a, ">", b)), then_b, else_b)]
    result = _normalize_hir_list(nodes)
    node = result[0]
    assert isinstance(node.condition, BinOp)
    assert node.condition.op == "<"
    assert node.condition.lhs == b
    assert node.condition.rhs == a
    assert node.then_nodes is else_b
    assert node.else_nodes is then_b


# ── Recursive normalization ───────────────────────────────────────────────────

def test_nested_if_normalized():
    """Nested IfNode inside then_nodes is normalized too."""
    inner_cond = BinOp(Reg("R1"), ">", Const(3))
    inner = _if(inner_cond, _body())
    outer_cond = BinOp(Reg("A"), "==", Const(0))
    nodes = [_if(outer_cond, [inner])]
    result = _normalize_hir_list(nodes)
    outer = result[0]
    inner_result = outer.then_nodes[0]
    assert isinstance(inner_result, IfNode)
    assert inner_result.condition.op == "<"
    assert inner_result.condition.lhs == Const(3)
    assert inner_result.condition.rhs == Reg("R1")


def test_non_if_nodes_unchanged():
    """Assign nodes pass through unchanged."""
    assign = Assign(0x1000, Reg("A"), Reg("R0"))
    nodes = [assign]
    result = _normalize_hir_list(nodes)
    assert result is nodes   # identity: no change → same list returned


# ── Integration: post_run via IfElseStructurer ────────────────────────────────

def test_post_run_normalizes_block_hir():
    """IfElseStructurer.post_run() normalizes conditions in each block's HIR."""
    from pseudo8051.passes.ifelse import IfElseStructurer

    cond = BinOp(Reg("A"), ">", Const(5))
    block = FakeBlock(0x1000, hir=[_if(cond, _body())])
    func  = FakeFunction("f", [block])
    IfElseStructurer().post_run(func)
    node = block.hir[0]
    assert isinstance(node, IfNode)
    assert node.condition.op == "<"
    assert node.condition.lhs == Const(5)
    assert node.condition.rhs == Reg("A")
