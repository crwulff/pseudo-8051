"""
passes/cjne_switch.py — CJNEChainToSwitch: rewrite nested-if CJNE chains as SwitchNode.

The 8051 CJNE (Compare and Jump if Not Equal) instruction produces a chain of
blocks where each block starts with a comparison against the next constant, jumping
past the case body if not equal.  After IfElseStructurer this appears as deeply
nested IfNodes of the form:

    if (reg != N) {
        [reg = reload;]        ← optional reload preamble (A = R7 etc.)
        if (reg != M) { … } else { case_M_body }
    } else { case_N_body }

Each then_nodes list may begin with one or more register-reload assignments
(Assign(comparison_reg, …)) before the next IfNode — these are discarded when
building the switch since the switch subject makes them redundant.

Termination cases:
  • then_nodes (after skipping preamble) is empty or non-IfNode  → default_body
  • tail IfNode has condition !(reg != val) or reg == val, no else → final case

Runs after func.hir is assembled (post-SwitchBodyAbsorber) and before
TypeAwareSimplifier, so that subsequent type-propagation sees the switch.
"""

from typing import List, Optional, Tuple

from pseudo8051.passes  import OptimizationPass
from pseudo8051.ir.hir  import HIRNode, IfNode, SwitchNode
from pseudo8051.ir.hir.assign      import Assign
from pseudo8051.ir.hir.break_stmt  import BreakStmt
from pseudo8051.ir.hir.return_stmt import ReturnStmt
from pseudo8051.ir.hir.goto_statement import GotoStatement
from pseudo8051.ir.expr import Regs as RegsExpr, Const as ConstExpr, BinOp, UnaryOp

_MIN_CASES = 3   # minimum distinct case values required to rewrite as switch

_TERMINATORS = (BreakStmt, ReturnStmt, GotoStatement)


def _ensure_break(body: List[HIRNode], ea: int) -> List[HIRNode]:
    """Return body with a BreakStmt appended if it doesn't already end with a terminator."""
    if body and isinstance(body[-1], _TERMINATORS):
        return body
    return body + [BreakStmt(ea)]


# ── Condition helpers ─────────────────────────────────────────────────────────

def _extract_ne(cond) -> Optional[Tuple[RegsExpr, int]]:
    """If cond is (single_reg != const), return (reg_expr, value). Else None."""
    if not isinstance(cond, BinOp) or cond.op != '!=':
        return None
    lhs, rhs = cond.lhs, cond.rhs
    if isinstance(lhs, RegsExpr) and lhs.is_single and isinstance(rhs, ConstExpr):
        return lhs, rhs.value
    if isinstance(rhs, RegsExpr) and rhs.is_single and isinstance(lhs, ConstExpr):
        return rhs, lhs.value
    return None


def _extract_eq(cond) -> Optional[Tuple[RegsExpr, int]]:
    """If cond is (single_reg == const) or !(single_reg != const), return (reg_expr, value)."""
    if isinstance(cond, BinOp) and cond.op == '==':
        lhs, rhs = cond.lhs, cond.rhs
        if isinstance(lhs, RegsExpr) and lhs.is_single and isinstance(rhs, ConstExpr):
            return lhs, rhs.value
        if isinstance(rhs, RegsExpr) and rhs.is_single and isinstance(lhs, ConstExpr):
            return rhs, lhs.value
    # !(reg != val)  — IfElseStructurer sometimes emits this form
    if isinstance(cond, UnaryOp) and cond.op == '!':
        return _extract_ne(cond.operand)
    return None


# ── Preamble helper ───────────────────────────────────────────────────────────

def _tail_if(then: List[HIRNode], reg_name: str) -> Optional[IfNode]:
    """Return the last node of then if it is an IfNode, provided all preceding
    nodes are Assign nodes that write the comparison register (reload preamble).
    Returns None otherwise.
    """
    if not then or not isinstance(then[-1], IfNode):
        return None
    for node in then[:-1]:
        if not (isinstance(node, Assign)
                and isinstance(node.lhs, RegsExpr)
                and node.lhs.is_single
                and node.lhs.name == reg_name):
            return None
    return then[-1]


# ── Chain detection ───────────────────────────────────────────────────────────

def _detect_chain(
        node: HIRNode,
) -> Optional[Tuple[RegsExpr, List[Tuple[int, List[HIRNode]]], Optional[List[HIRNode]]]]:
    """Detect a CJNE chain rooted at node.

    Returns (reg_expr, [(val, body), ...], default_body) where:
      • reg_expr    — the single register being compared throughout
      • cases       — (value, else_body) pairs, outermost first
      • default_body — the unmatched path (then_nodes of innermost !=), or None

    Returns None if node is not the head of a qualifying chain.
    """
    if not isinstance(node, IfNode):
        return None
    if _extract_ne(node.condition) is None:
        return None

    reg_name: str = _extract_ne(node.condition)[0].name
    reg_expr: RegsExpr = _extract_ne(node.condition)[0]
    cases: List[Tuple[int, List[HIRNode]]] = []
    current: IfNode = node

    while True:
        ne = _extract_ne(current.condition)
        if ne is None or ne[0].name != reg_name:
            break

        reg_expr = ne[0]
        val      = ne[1]
        cases.append((val, list(current.else_nodes)))
        then = current.then_nodes

        # Find the tail IfNode (past any reload preamble)
        tail = _tail_if(then, reg_name)

        # Equality-check terminator: if (reg == val) { body } with no else
        if tail is not None and not tail.else_nodes:
            eq = _extract_eq(tail.condition)
            if eq is not None and eq[0].name == reg_name:
                cases.append((eq[1], list(tail.then_nodes)))
                return (reg_expr, cases, None)

        # Continue chain: tail IfNode continues with != on same register
        if tail is not None:
            inner_ne = _extract_ne(tail.condition)
            if inner_ne is not None and inner_ne[0].name == reg_name:
                current = tail
                continue

        # Terminate: remaining then_nodes = default body
        default_body = list(then) if then else None
        return (reg_expr, cases, default_body)

    return None


# ── Node-list transformer ─────────────────────────────────────────────────────

def _transform_nodes(nodes: List[HIRNode]) -> List[HIRNode]:
    """Top-down transformation: detect the full chain first, then recurse into bodies.

    Top-down is critical: bottom-up would convert inner 3-case sub-chains before
    the outer chain detection runs, breaking the larger chain at those points.
    """
    result: List[HIRNode] = []
    for node in nodes:
        chain = _detect_chain(node)
        if chain is not None and len(chain[1]) >= _MIN_CASES:
            reg_expr, cases, default_body = chain
            # Recurse into case bodies AFTER detection; add break unless body
            # already ends with a flow-control terminator.
            sw_cases = [([val], _ensure_break(_transform_nodes(body), node.ea))
                        for val, body in cases]
            sw_default = _transform_nodes(default_body) if default_body else None
            sw = SwitchNode(node.ea, reg_expr, sw_cases, default_body=sw_default)
            node.copy_meta_to(sw)
            result.append(sw)
        else:
            # No chain here — recurse into children normally
            result.append(node.map_bodies(_transform_nodes))
    return result


# ── Pass ──────────────────────────────────────────────────────────────────────

class CJNEChainToSwitch(OptimizationPass):
    """Rewrite nested-if CJNE chains in func.hir as SwitchNodes."""

    def run(self, func) -> None:
        func.hir = _transform_nodes(func.hir)
        from pseudo8051.constants import DEBUG
        if DEBUG:
            from pseudo8051.passes.debug_dump import dump_pass_hir
            dump_pass_hir("08b.cjne_switch", func.hir, func.name)
