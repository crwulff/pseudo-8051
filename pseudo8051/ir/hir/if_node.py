"""
ir/hir/if_node.py — IfNode structured control-flow node.
"""

from typing import Callable, List, Optional, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_cond, _ann_field, _Cond


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

    def map_bodies(self, fn: Callable[[List[HIRNode]], List[HIRNode]]) -> "IfNode":
        return IfNode(self.ea, self.condition, fn(self.then_nodes), fn(self.else_nodes))

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

    def ann_lines(self) -> List[str]:
        return (["IfNode"] + _ann_field("cond", self.condition)
                + [f"  then: {len(self.then_nodes)} nodes",
                   f"  else: {len(self.else_nodes)} nodes"])
