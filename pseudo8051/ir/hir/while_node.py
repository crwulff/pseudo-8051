"""
ir/hir/while_node.py — WhileNode structured control-flow node.
"""

from typing import Callable, List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_cond, _ann_field, _Cond, _killed_by_seq, _possibly_killed_by_seq, _cond_refs


class WhileNode(HIRNode):
    """
    while (condition) { body_nodes }
    """

    def __init__(self, ea: int, condition: _Cond,
                 body_nodes: List[HIRNode]):
        super().__init__(ea)
        self.condition  = condition
        self.body_nodes = body_nodes

    def child_body_groups(self):
        return [(0, self.body_nodes)]

    def map_bodies(self, fn: Callable[[List[HIRNode]], List[HIRNode]]) -> "WhileNode":
        return self.copy_meta_to(WhileNode(self.ea, self.condition, fn(self.body_nodes)))

    def definitely_killed(self) -> frozenset:
        """Body may execute zero times, so no register is guaranteed killed."""
        return frozenset()

    def possibly_killed(self) -> frozenset:
        """Registers possibly killed if the body executes at least once."""
        return _possibly_killed_by_seq(self.body_nodes)

    def replace_condition(self, new_cond) -> "WhileNode":
        return self.copy_meta_to(WhileNode(self.ea, new_cond, self.body_nodes))

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}while ({_render_cond(self.condition)}) {{"))
        for node in self.body_nodes:
            lines.extend(node.render(indent + 1))
        lines.append((self.ea, f"{ind}}}"))
        return lines

    def name_refs(self) -> frozenset:
        body_refs = frozenset().union(*(n.name_refs() for n in self.body_nodes))
        return _cond_refs(self.condition) | body_refs

    def ann_lines(self) -> List[str]:
        return (["WhileNode"] + _ann_field("cond", self.condition)
                + [f"  body: {len(self.body_nodes)} nodes"])
