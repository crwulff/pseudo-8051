"""
ir/hir/do_while_node.py — DoWhileNode structured control-flow node.
"""

from typing import Callable, List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_cond, _ann_field, _Cond


class DoWhileNode(HIRNode):
    """
    do { body_nodes } while (condition);

    condition may be str (legacy) or Expr.
    """

    def __init__(self, ea: int, condition: _Cond, body_nodes: List[HIRNode]):
        super().__init__(ea)
        self.condition  = condition
        self.body_nodes = body_nodes

    def map_bodies(self, fn: Callable[[List[HIRNode]], List[HIRNode]]) -> "DoWhileNode":
        return DoWhileNode(self.ea, self.condition, fn(self.body_nodes))

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}do {{"))
        for node in self.body_nodes:
            lines.extend(node.render(indent + 1))
        lines.append((self.ea, f"{ind}}} while ({_render_cond(self.condition)});"))
        return lines

    def name_refs(self) -> frozenset:
        return frozenset().union(*(n.name_refs() for n in self.body_nodes))

    def ann_lines(self) -> List[str]:
        return (["DoWhileNode"] + _ann_field("cond", self.condition)
                + [f"  body: {len(self.body_nodes)} nodes"])
