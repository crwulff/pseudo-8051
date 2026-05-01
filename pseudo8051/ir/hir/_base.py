"""
ir/hir/_base.py — NodeAnnotation, HIRNode ABC, and shared helpers.
"""

import os as _os
import sys as _sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, FrozenSet, List, Optional, Tuple, Union

from pseudo8051.ir.expr import Expr, Regs as _RegsExpr, Name as _NameExpr

# Directory containing this file (ir/hir/); used to skip subclass __init__ frames.
_HIR_DIR = _os.path.normpath(_os.path.abspath(_os.path.dirname(__file__))) + _os.sep


# Ordered list of (path-substring, label) pairs.  First match wins.
# More-specific entries must precede less-specific ones.
_PASS_NAME_TABLE = [
    # typesimplify sub-modules (specific before generic)
    ("typesimplify/_propagate",   "Propagation"),
    ("typesimplify/_setup_fold",  "SetupFold"),
    ("typesimplify/_simplify",    "Simplifier"),
    ("typesimplify/_post",        "PostProcess"),
    ("typesimplify/_return_fold", "ReturnFold"),
    ("typesimplify/",             "Simplifier"),
    # handlers
    ("handlers/",                 "Lift"),
    # pass modules (split sub-modules listed before the public entry point)
    ("passes/annotate",           "Annotation"),
    ("passes/rmw",                "RMW"),
    ("passes/chunk_inline",       "ChunkInline"),
    ("passes/_switch_detect",     "Switch"),
    ("passes/_switch_build",      "Switch"),
    ("passes/switch",             "Switch"),
    ("passes/_ifelse_helpers",    "IfElse"),
    ("passes/ifelse",             "IfElse"),
    ("passes/jmptable",           "JmpTable"),
    # IR construction
    ("ir/basicblock",             "InitialHIR"),
    ("ir/function",               "Function"),
]


def _pass_name(short: str) -> str:
    """Derive a human-readable pass label from a pseudo8051-relative file path."""
    for key, label in _PASS_NAME_TABLE:
        if key in short:
            return label
    if "passes/patterns/" in short:
        base = short.rsplit("/", 1)[-1].replace(".py", "")
        return f"Pattern:{base}"
    return ""


def _node_creator() -> str:
    """Return 'pass @ pseudo8051/…/file.py:lineno' for the first call-stack frame outside ir/hir/.

    Called from HIRNode.__init__; walks up past any subclass __init__ chains that
    live inside ir/hir/ to find the first external creator (handler, pattern, pass, …).
    """
    frame = _sys._getframe(2)   # skip _node_creator → HIRNode.__init__ → start here
    while frame is not None:
        fn = _os.path.normpath(_os.path.abspath(frame.f_code.co_filename)) + _os.sep
        if not fn.startswith(_HIR_DIR):
            break
        frame = frame.f_back
    if frame is None:
        return "unknown"
    fn = frame.f_code.co_filename.replace("\\", "/")
    idx = fn.rfind("pseudo8051/")
    short = fn[idx:] if idx >= 0 else _os.path.basename(fn)
    loc = f"{short}:{frame.f_lineno}"
    label = _pass_name(short)
    return f"{label} @ {loc}" if label else loc

if TYPE_CHECKING:
    from pseudo8051.passes.patterns._utils import VarInfo, TypeGroup


class NodeAnnotation:
    """Per-node annotation: register names/types and constant values at this point."""
    __slots__ = ("reg_groups", "reg_consts", "reg_exprs", "reg_expr_sources",
                 "call_arg_ann", "callee_args", "user_anns")

    def __init__(self):
        self.reg_groups:      "List[TypeGroup]"           = []   # forward-propagated TypeGroups
        self.reg_consts:      Dict[str, int]               = {}   # reg → known const
        self.reg_exprs:       "Dict[str, Expr]"            = {}   # reg → defining Expr (met across preds)
        self.reg_expr_sources: "Dict[str, HIRNode]"        = {}   # reg → HIR node that last defined it
        self.call_arg_ann:    "List[TypeGroup]"            = []   # backward-propagated callee params
        self.callee_args:     "Optional[List[TypeGroup]]"  = None # call node only
        self.user_anns:       "List[TypeGroup]"            = []   # user register annotations (force-installed)

    @staticmethod
    def merge(first: "object", last: "object") -> "Optional[NodeAnnotation]":
        """Build an annotation for a node that replaces a span of consumed nodes.

        *first* and *last* may be HIRNode objects or NodeAnnotation objects (or None).

        reg_groups / reg_consts come from *first* (the state at the start of the
        span).  call_arg_ann comes from *last* (what downstream code expects after
        the span).  callee_args is taken from whichever side has it (first wins).
        """
        first_ann = (first.ann if hasattr(first, 'ann') else first)
        last_ann  = (last.ann  if hasattr(last,  'ann') else last)
        if first_ann is None and last_ann is None:
            return None
        ann = NodeAnnotation()
        if first_ann is not None:
            ann.reg_groups        = first_ann.reg_groups
            ann.reg_consts        = first_ann.reg_consts
            ann.reg_exprs         = first_ann.reg_exprs
            ann.reg_expr_sources  = first_ann.reg_expr_sources
            ann.callee_args       = first_ann.callee_args
        if last_ann is not None:
            ann.call_arg_ann = last_ann.call_arg_ann
            if ann.callee_args is None:
                ann.callee_args = last_ann.callee_args
            ann.user_anns = ann.user_anns + last_ann.user_anns
        return ann

    # ── New TypeGroup lookup helpers ──────────────────────────────────────────

    def group_for(self, reg: str) -> "Optional[TypeGroup]":
        """Find the TypeGroup whose active_regs contains *reg*."""
        for g in self.reg_groups:
            if reg in g.active_regs:
                return g
        return None

    def call_arg_for(self, reg: str) -> "Optional[TypeGroup]":
        """Find the call_arg_ann TypeGroup whose active_regs contains *reg*."""
        for g in self.call_arg_ann:
            if reg in g.active_regs:
                return g
        return None



@dataclass
class RemovedNode:
    """Record of a HIR node eliminated by a transform, with the reason for removal."""
    node:   "HIRNode"
    reason: str


class HIRNode(ABC):
    """Abstract base for all HIR nodes."""

    def __init__(self, ea: int):
        self.ea           = ea
        self.source_nodes: List["HIRNode"] = []   # immediate inputs; empty = leaf from IDA
        self.ann: Optional[NodeAnnotation] = None
        self._creator: str = _node_creator()

    @property
    def src_eas(self) -> frozenset:
        """Recursively union EAs from all source nodes; leaf nodes return {self.ea}."""
        return self._src_eas_inner(set())

    def _src_eas_inner(self, seen: set) -> frozenset:
        if id(self) in seen:
            return frozenset()
        seen.add(id(self))
        if not self.source_nodes:
            return frozenset({self.ea})
        result: frozenset = frozenset()
        for sn in self.source_nodes:
            result |= sn._src_eas_inner(seen)
        return result

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

    def copy_meta_to(self, dst: "HIRNode") -> "HIRNode":
        """Copy ann and source_nodes from self to dst; return dst for chaining.

        Use this whenever a new node is constructed to replace an existing one
        and provenance should be preserved unchanged:
            node = node.copy_meta_to(Assign(node.ea, new_lhs, new_rhs))
        """
        dst.ann          = self.ann
        dst.source_nodes = self.source_nodes
        return dst

    def child_body_groups(self) -> "List[Tuple[int, List['HIRNode']]]":
        """Return the ordered list of child body lists for line-map building.

        Each entry is (extra_header_lines, body_nodes) where extra_header_lines
        is the number of structural lines to skip before this body begins
        (0 for the first body, 1 for '} else {', etc.).

        Leaf nodes return [].  SwitchNode also returns [] — it is handled
        specially by the viewer because it needs the _SwitchCaseView sentinel.
        """
        return []

    def map_bodies(self, fn: "Callable[[List[HIRNode]], List[HIRNode]]") -> "HIRNode":
        """Return a copy of this node with fn applied to every child body list.

        Leaf nodes (Assign, ExprStmt, ReturnStmt, etc.) return self unchanged.
        Structured nodes (IfNode, WhileNode, etc.) rebuild with mapped bodies.
        """
        return self

    def map_exprs(self, fn: "Callable[[Expr], Expr]") -> "HIRNode":
        """Return a copy of this node with fn applied to every read-position Expr.

        LHS of assignments is not transformed (it is a write destination, not a
        read position).  Returns self unchanged if fn changes nothing.
        Leaf-like nodes with no expression operands return self.

        Callers that need to track provenance should copy metadata after calling
        this method (map_exprs does not touch ann or source_nodes).
        """
        return self

    def ann_lines(self) -> List[str]:
        """Return annotation lines for this node (class name + fields)."""
        return [type(self).__name__]

    def node_ann_lines(self) -> List[str]:
        """Return annotation lines for self.ann (NodeAnnotation fields), or [] if absent."""
        if self.ann is None:
            return []
        out: List[str] = []
        ann = self.ann
        if ann.reg_groups:
            out.append("  ann.reg_groups:")
            for g in ann.reg_groups:
                active = ",".join(sorted(g.active_regs, key=lambda r: g.full_regs.index(r)
                                         if r in g.full_regs else 99))
                full   = ",".join(g.full_regs)
                regs   = active if active == full else f"{active}/{full}"
                label  = f"param " if g.is_param else ""
                xram   = f" @{g.xram_sym}" if g.xram_sym else ""
                out.append(f"    {label}{g.name}: {g.type} [{regs}]{xram}")
        if ann.reg_consts:
            out.append("  ann.reg_consts: " +
                       ", ".join(f"{k}={v:#x}" for k, v in sorted(ann.reg_consts.items())))
        if ann.reg_exprs:
            out.append("  ann.reg_exprs: " +
                       ", ".join(f"{k}={v.render()}" for k, v in sorted(ann.reg_exprs.items())))
        if ann.call_arg_ann:
            out.append("  ann.call_arg_ann:")
            for g in ann.call_arg_ann:
                active = ",".join(sorted(g.active_regs, key=lambda r: g.full_regs.index(r)
                                          if r in g.full_regs else 99))
                full   = ",".join(g.full_regs)
                regs   = active if active == full else f"{active}/{full}"
                xram   = f" @{g.xram_sym}" if g.xram_sym else ""
                out.append(f"    {g.name}: {g.type} [{regs}]{xram}")
        if ann.callee_args is not None:
            out.append("  ann.callee_args:")
            for g in ann.callee_args:
                active = ",".join(sorted(g.active_regs, key=lambda r: g.full_regs.index(r)
                                          if r in g.full_regs else 99))
                full   = ",".join(g.full_regs)
                regs   = active if active == full else f"{active}/{full}"
                label  = "param " if g.is_param else "retval "
                out.append(f"    {label}{g.name}: {g.type} [{regs}]")
        return out

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
        return expr.reg_set()
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


def _render_call_with_comments(call_expr: "Expr", ann: "Optional[object]") -> str:
    """Render a Call expression, annotating each arg with the callee param name.

    If ann.callee_args provides names for the call's positional arguments,
    renders as: func(arg0 /* param0 */, arg1 /* param1 */, ...)
    Falls back to plain rendering when no annotation is available.
    """
    from pseudo8051.ir.expr import Call as _Call
    if not isinstance(call_expr, _Call):
        return _render_expr(call_expr)

    callee_args = getattr(ann, "callee_args", None) if ann is not None else None
    if not callee_args:
        return call_expr.render()

    # Build ordered list of param names from callee_args TypeGroups.
    # callee_args is ordered by the prototype parameter list.
    # Exclude byte-field expansions (e.g. offset.hi, offset.lo) and return-value
    # TypeGroups (is_param=False) — only top-level parameters should appear.
    param_names = [tg.name for tg in callee_args
                   if tg.is_param and not getattr(tg, 'is_byte_field', False)]

    parts = []
    for i, arg in enumerate(call_expr.args):
        rendered = arg.render()
        if i < len(param_names):
            name = param_names[i]
            # Suppress comment when the rendered arg already contains the param name
            # (e.g. Name("glyphIndex") renders as "glyphIndex" — no comment needed).
            if name and name not in rendered:
                rendered = f"{rendered} /* {name} */"
        parts.append(rendered)

    # Append placeholder comments for any expected params beyond the actual arg list
    # (xram params that weren't folded in because they were set in an outer scope).
    for i in range(len(call_expr.args), len(param_names)):
        name = param_names[i]
        if name:
            parts.append(f"/* {name} */")

    return f"{call_expr.func_name}({', '.join(parts)})"


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
        return lhs.reg_set()
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


def _possibly_killed_by_seq(nodes: "List[HIRNode]") -> frozenset:
    """Union of registers possibly killed by a sequential list of nodes.

    A register is possibly killed by a sequence if ANY node in the sequence
    possibly kills it.  For structured nodes this recurses through their bodies
    (via possibly_killed()), unlike _killed_by_seq which uses definitely_killed().
    Used by structured-node possibly_killed() overrides.
    """
    result: frozenset = frozenset()
    for n in nodes:
        result |= n.possibly_killed()
    return result


# ── Condition type alias ──────────────────────────────────────────────────────

_Cond = Expr


def _render_cond(c: _Cond) -> str:
    """Render a condition Expr to a string."""
    return c.render()


def _cond_refs(c: _Cond) -> frozenset:
    """Return register/name refs from a condition Expr."""
    return _refs_from_expr(c)
