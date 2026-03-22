"""
ir/hir.py — High-level IR nodes.

All nodes carry an ea (source address) for viewer double-click navigation.
render() returns a list of (ea, indented_text) tuples.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from pseudo8051.ir.expr import Expr


class HIRNode(ABC):
    """Abstract base for all HIR nodes."""

    def __init__(self, ea: int):
        self.ea = ea

    @abstractmethod
    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        """Return list of (ea, text) tuples at the given indent level."""
        ...

    @staticmethod
    def _ind(indent: int) -> str:
        return "    " * indent


# ── Expression-tree statement nodes ───────────────────────────────────────────

def _render_expr(val: Union[str, Expr]) -> str:
    """Render either a plain string or an Expr to a string."""
    if isinstance(val, Expr):
        return val.render()
    return str(val)


class Assign(HIRNode):
    """lhs = rhs;"""

    def __init__(self, ea: int, lhs: Expr, rhs: Expr):
        super().__init__(ea)
        self.lhs = lhs
        self.rhs = rhs

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{_render_expr(self.lhs)} = {_render_expr(self.rhs)};")]


class CompoundAssign(HIRNode):
    """lhs op= rhs;  e.g. A += rhs;"""

    def __init__(self, ea: int, lhs: Expr, op: str, rhs: Expr):
        super().__init__(ea)
        self.lhs = lhs
        self.op  = op
        self.rhs = rhs

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{_render_expr(self.lhs)} {self.op} {_render_expr(self.rhs)};")]


class ExprStmt(HIRNode):
    """A standalone expression statement: push(R7);  R7++;"""

    def __init__(self, ea: int, expr: Expr):
        super().__init__(ea)
        self.expr = expr

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{_render_expr(self.expr)};")]


class ReturnStmt(HIRNode):
    """return;  or  return expr;"""

    def __init__(self, ea: int, value: Optional[Expr] = None):
        super().__init__(ea)
        self.value = value

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        if self.value is None:
            return [(self.ea, f"{self._ind(indent)}return;")]
        return [(self.ea, f"{self._ind(indent)}return {_render_expr(self.value)};")]


class IfGoto(HIRNode):
    """if (cond) goto label;"""

    def __init__(self, ea: int, cond: Expr, label: str):
        super().__init__(ea)
        self.cond  = cond
        self.label = label

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}if ({_render_expr(self.cond)}) goto {self.label};")]


# ── Legacy string-based statement (kept for test compat; see Phase 8 note) ───

class Statement(HIRNode):
    """A single C-like statement string (already formatted by a handler)."""

    def __init__(self, ea: int, text: str):
        super().__init__(ea)
        self.text = text

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{self.text}")]


# ── Flow control ──────────────────────────────────────────────────────────────

class GotoStatement(HIRNode):
    """goto label;"""

    def __init__(self, ea: int, label: str):
        super().__init__(ea)
        self.label = label

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}goto {self.label};")]


class Label(HIRNode):
    """label_XXXX: — emitted before a block that needs a label."""

    def __init__(self, ea: int, name: str):
        super().__init__(ea)
        self.name = name

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [
            (self.ea, ""),
            (self.ea, f"{self.name}:"),
        ]


# ── Condition type alias ──────────────────────────────────────────────────────
# Structural nodes accept str | Expr during migration; Phase 8 removes str.

_Cond = Union[str, Expr]


def _render_cond(c: _Cond) -> str:
    """Render a condition that is either a plain str or an Expr."""
    if isinstance(c, Expr):
        return c.render()
    return str(c)


# ── Structured control-flow nodes ─────────────────────────────────────────────

class IfNode(HIRNode):
    """
    if (condition) { then_nodes } [else { else_nodes }]

    condition may be str (legacy) or Expr (Phase 7+).
    """

    def __init__(self, ea: int, condition: _Cond,
                 then_nodes: List[HIRNode],
                 else_nodes: Optional[List[HIRNode]] = None):
        super().__init__(ea)
        self.condition  = condition
        self.then_nodes = then_nodes
        self.else_nodes = else_nodes or []

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}if ({_render_cond(self.condition)}) {{"))
        for node in self.then_nodes:
            lines.extend(node.render(indent + 1))
        if self.else_nodes:
            lines.append((self.ea, f"{ind}}} else {{"))
            for node in self.else_nodes:
                lines.extend(node.render(indent + 1))
        lines.append((self.ea, f"{ind}}}"))
        return lines


class WhileNode(HIRNode):
    """
    while (condition) { body_nodes }

    condition may be str (legacy) or Expr (Phase 7+).
    """

    def __init__(self, ea: int, condition: _Cond,
                 body_nodes: List[HIRNode]):
        super().__init__(ea)
        self.condition  = condition
        self.body_nodes = body_nodes

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}while ({_render_cond(self.condition)}) {{"))
        for node in self.body_nodes:
            lines.extend(node.render(indent + 1))
        lines.append((self.ea, f"{ind}}}"))
        return lines


class ForNode(HIRNode):
    """
    for (init; condition; update) { body_nodes }

    init/condition/update may be str (legacy) or Assign/Expr (Phase 7+).
    """

    def __init__(self, ea: int,
                 init: Union[str, Expr, Assign],
                 condition: _Cond,
                 update: Union[str, Expr],
                 body_nodes: List[HIRNode]):
        super().__init__(ea)
        self.init       = init
        self.condition  = condition
        self.update     = update
        self.body_nodes = body_nodes

    def _render_init(self) -> str:
        """Render the for-loop init clause (no trailing semicolon)."""
        if isinstance(self.init, Assign):
            return f"{_render_expr(self.init.lhs)} = {_render_expr(self.init.rhs)}"
        if isinstance(self.init, Expr):
            return self.init.render()
        return str(self.init)

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea,
                       f"{ind}for ({self._render_init()}; "
                       f"{_render_cond(self.condition)}; "
                       f"{_render_cond(self.update)}) {{"))
        for node in self.body_nodes:
            lines.extend(node.render(indent + 1))
        lines.append((self.ea, f"{ind}}}"))
        return lines


class SwitchNode(HIRNode):
    """
    switch (subject) {
        case 2: goto label_a;
        case 4: case 8: goto label_b;
        default: goto label_default;
    }

    cases is a list of (values_list, body) pairs where body is either:
      - str: goto label (pre-absorption)
      - List[HIRNode]: inlined body (post-absorption by SwitchBodyAbsorber)

    default_label is the target for unmatched values (from a trailing jnz), or None.
    default_body is the inlined default body (post-absorption), or None.
    """

    def __init__(self, ea: int, subject: Expr,
                 cases: List[Tuple[List[int], Union[str, List['HIRNode']]]],
                 default_label: Optional[str] = None,
                 default_body: Optional[List['HIRNode']] = None):
        super().__init__(ea)
        self.subject       = subject
        self.cases         = cases
        self.default_label = default_label
        self.default_body  = default_body

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind  = self._ind(indent)
        ind1 = self._ind(indent + 1)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}switch ({_render_expr(self.subject)}) {{"))
        for values, body in self.cases:
            case_prefix = " ".join(f"case {v}:" for v in values)
            if isinstance(body, str):
                lines.append((self.ea, f"{ind1}{case_prefix} goto {body};"))
            else:
                lines.append((self.ea, f"{ind1}{case_prefix}"))
                for node in body:
                    lines.extend(node.render(indent + 2))
        if self.default_body is not None:
            lines.append((self.ea, f"{ind1}default:"))
            for node in self.default_body:
                lines.extend(node.render(indent + 2))
        elif self.default_label is not None:
            lines.append((self.ea, f"{ind1}default: goto {self.default_label};"))
        lines.append((self.ea, f"{ind}}}"))
        return lines
