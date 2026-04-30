"""
ir/hir/do_while_node.py — DoWhileNode structured control-flow node.
"""

from typing import Callable, List, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_cond, _ann_field, _Cond, _killed_by_seq, _possibly_killed_by_seq, _cond_refs


class DoWhileNode(HIRNode):
    """
    do { body_nodes } while (condition);
    """

    def __init__(self, ea: int, condition: _Cond, body_nodes: List[HIRNode]):
        super().__init__(ea)
        self.condition  = condition
        self.body_nodes = body_nodes

    def child_body_groups(self):
        return [(0, self.body_nodes)]

    def map_bodies(self, fn: Callable[[List[HIRNode]], List[HIRNode]]) -> "DoWhileNode":
        return self.copy_meta_to(DoWhileNode(self.ea, self.condition, fn(self.body_nodes)))

    def definitely_killed(self) -> frozenset:
        """Body executes at least once, so all sequential writes are definite kills."""
        return _killed_by_seq(self.body_nodes)

    def possibly_killed(self) -> frozenset:
        return _possibly_killed_by_seq(self.body_nodes)

    def replace_condition(self, new_cond) -> "DoWhileNode":
        return self.copy_meta_to(DoWhileNode(self.ea, new_cond, self.body_nodes))

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}do {{"))
        for node in self.body_nodes:
            lines.extend(node.render(indent + 1))
        lines.append((self.ea, f"{ind}}} while ({_render_cond(self.condition)});"))
        return lines

    def name_refs(self) -> frozenset:
        body_refs = frozenset().union(*(n.name_refs() for n in self.body_nodes))
        return _cond_refs(self.condition) | body_refs

    def ann_lines(self) -> List[str]:
        return (["DoWhileNode"] + _ann_field("cond", self.condition)
                + [f"  body: {len(self.body_nodes)} nodes"])
