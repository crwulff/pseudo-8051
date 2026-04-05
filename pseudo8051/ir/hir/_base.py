"""
ir/hir/_base.py — NodeAnnotation, HIRNode ABC, and shared helpers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from pseudo8051.ir.expr import Expr, Regs as _RegsExpr, Name as _NameExpr

if TYPE_CHECKING:
    from pseudo8051.passes.patterns._utils import VarInfo


class NodeAnnotation:
    """Per-node annotation: register names/types and constant values at this point."""
    __slots__ = ("reg_names", "reg_consts", "call_arg_ann", "callee_args")

    def __init__(self):
        self.reg_names:    "Dict[str, VarInfo]" = {}   # reg → VarInfo (name, type)
        self.reg_consts:   Dict[str, int]        = {}   # reg → known const
        self.call_arg_ann: "Dict[str, VarInfo]"  = {}   # backward-propagated callee param
        self.callee_args:  "Optional[Dict[str, VarInfo]]" = None  # call node only


class HIRNode(ABC):
    """Abstract base for all HIR nodes."""

    def __init__(self, ea: int):
        self.ea  = ea
        self.ann: Optional[NodeAnnotation] = None

    @abstractmethod
    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        """Return list of (ea, text) tuples at the given indent level."""
        ...

    @property
    def written_regs(self) -> frozenset:
        """Register/name strings written (defined) as the primary LHS of this node.

        For RegGroup LHS includes both individual names and the pair key
        (e.g. {"R6", "R7", "R6R7"}).  Returns frozenset() for nodes that
        write no registers (structured nodes, ReturnStmt, etc.).
        """
        return frozenset()

    @property
    def def_regs(self) -> frozenset:
        """Registers definitely written directly by this node (non-recursive).

        Alias for written_regs provided for the new propagation interface.
        Structured nodes return frozenset() — their bodies' writes are
        reported via definitely_killed() / possibly_killed().
        """
        return self.written_regs

    @property
    def use_regs(self) -> frozenset:
        """Register names read by this node (non-recursive).

        Returns frozenset() by default; overridden by statement nodes that
        read registers in their expressions.
        """
        return frozenset()

    def definitely_killed(self) -> frozenset:
        """Registers killed on ALL execution paths through this node's bodies.

        For leaf nodes this equals def_regs (the single write is always
        definite).  Structured nodes override to compute the intersection
        of their branch kills; loop nodes return frozenset() because the
        body may execute zero times.
        """
        return self.def_regs

    def possibly_killed(self) -> frozenset:
        """Registers killed on ANY execution path through this node's bodies.

        For leaf nodes this equals def_regs.  Structured nodes override to
        compute the union of their branch kills.
        """
        return self.def_regs

    def map_bodies(self, fn: "Callable[[List[HIRNode]], List[HIRNode]]") -> "HIRNode":
        """Return a copy of this node with fn applied to every child body list.

        Leaf nodes (Assign, ExprStmt, ReturnStmt, etc.) return self unchanged.
        Structured nodes (IfNode, WhileNode, etc.) rebuild with mapped bodies.
        """
        return self

    def ann_lines(self) -> List[str]:
        """Return annotation lines for this node (class name + fields)."""
        return [type(self).__name__]

    def name_refs(self) -> frozenset:
        """Collect all Reg/Name/RegGroup strings referenced in read positions in this node."""
        return frozenset()

    @staticmethod
    def _ind(indent: int) -> str:
        return "    " * indent


# ── Expression reference collection ──────────────────────────────────────────

def _refs_from_expr(expr: Expr) -> frozenset:
    """Collect all Reg/Name/RegGroup name strings referenced in read positions in expr."""
    if isinstance(expr, _RegsExpr):
        if expr.is_single:
            return frozenset({expr.names[0]})
        return frozenset(expr.names) | frozenset({"".join(expr.names)})
    if isinstance(expr, _NameExpr):
        return frozenset({expr.name})
    children = expr.children()
    if not children:
        return frozenset()
    return frozenset().union(*(_refs_from_expr(c) for c in children))


# ── Expression rendering helper ───────────────────────────────────────────────

def _render_expr(val: Union[str, Expr]) -> str:
    """Render either a plain string or an Expr to a string."""
    if isinstance(val, Expr):
        return val.render()
    return str(val)


# ── HIR annotation helpers ────────────────────────────────────────────────────

def _expr_lines(expr: Expr, indent: str = "") -> List[str]:
    """Return annotation lines for one Expr node, recursing into children."""
    name = type(expr).__name__
    if name == "BinOp":
        out = [f"{indent}BinOp {expr.op!r}"]
        out += _expr_lines(expr.lhs, indent + "  ")
        out += _expr_lines(expr.rhs, indent + "  ")
    elif name == "UnaryOp":
        suffix = " (post)" if getattr(expr, "post", False) else ""
        out = [f"{indent}UnaryOp {expr.op!r}{suffix}"]
        out += _expr_lines(expr.operand, indent + "  ")
    elif name in ("XRAMRef", "IRAMRef", "CROMRef"):
        out = [f"{indent}{name}"]
        out += _expr_lines(expr.inner, indent + "  ")
    elif name == "Cast":
        out = [f"{indent}Cast {expr.type_str!r}"]
        out += _expr_lines(expr.inner, indent + "  ")
    elif name == "Call":
        out = [f"{indent}Call {expr.func_name!r}"]
        for arg in expr.args:
            out += _expr_lines(arg, indent + "  ")
    else:  # Reg, Const, Name, RegGroup — useful reprs
        out = [f"{indent}{repr(expr)}"]
    return out


def _ann_field(label: str, val) -> List[str]:
    """Format one HIR annotation field as indented lines."""
    if val is None:
        return []
    if isinstance(val, Expr):
        return [f"  {label}:"] + [f"    {ln}" for ln in _expr_lines(val)]
    # HIRNode (e.g. ForNode's Assign init): render to text
    render = getattr(val, "render", None)
    if render is not None:
        try:
            text = render(0)[0][1].strip()
            return [f"  {label}: {text!r}"]
        except (IndexError, TypeError):
            pass
    return [f"  {label}: {val!r}"]


# ── LHS written-register helper ───────────────────────────────────────────────

def _lhs_written_regs(lhs: Expr) -> frozenset:
    """Extract written register names from an assignment LHS expression."""
    if isinstance(lhs, _RegsExpr):
        if lhs.is_single:
            return frozenset({lhs.names[0]})
        return frozenset(lhs.names) | {"".join(lhs.names)}
    return frozenset()


# ── Propagation helpers ───────────────────────────────────────────────────────

def _killed_by_seq(nodes: "List[HIRNode]") -> frozenset:
    """Union of registers definitely killed by a sequential list of nodes.

    In a sequential execution, every write is guaranteed to execute, so the
    set of killed registers is the union of each node's definitely_killed().
    Used by structured-node overrides to compute branch kill sets.
    """
    result: frozenset = frozenset()
    for n in nodes:
        result |= n.def_regs | n.definitely_killed()
    return result


# ── Condition type alias ──────────────────────────────────────────────────────
# Structural nodes accept str | Expr during migration; Phase 8 removes str.

_Cond = Union[str, Expr]


def _render_cond(c: _Cond) -> str:
    """Render a condition that is either a plain str or an Expr."""
    if isinstance(c, Expr):
        return c.render()
    return str(c)
