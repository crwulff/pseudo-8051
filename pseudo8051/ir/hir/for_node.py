"""
ir/hir/for_node.py — ForNode structured control-flow node.
"""

from typing import Callable, List, Tuple, Union

from pseudo8051.ir.hir._base import HIRNode, _render_expr, _render_cond, _ann_field, _Cond
from pseudo8051.ir.hir.assign import Assign
from pseudo8051.ir.expr import Expr


class ForNode(HIRNode):
    """
    for (init; condition; update) { body_nodes }

    init/condition/update may be str (legacy) or Assign/Expr (Phase 7+).
    init may be None for promoted loops where the counter is already set.
    """

    def __init__(self, ea: int,
                 init: Union[str, Expr, Assign, None],
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
        if self.init is None:
            return ""
        if isinstance(self.init, Assign):
            return f"{_render_expr(self.init.lhs)} = {_render_expr(self.init.rhs)}"
        if isinstance(self.init, Expr):
            return self.init.render()
        return str(self.init)

    def map_bodies(self, fn: Callable[[List[HIRNode]], List[HIRNode]]) -> "ForNode":
        return ForNode(self.ea, self.init, self.condition, self.update, fn(self.body_nodes))

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

    def name_refs(self) -> frozenset:
        return frozenset().union(*(n.name_refs() for n in self.body_nodes))

    def ann_lines(self) -> List[str]:
        return (["ForNode"] + _ann_field("init", self.init)
                + _ann_field("cond", self.condition)
                + _ann_field("update", self.update)
                + [f"  body: {len(self.body_nodes)} nodes"])
