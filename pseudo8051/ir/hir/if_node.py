"""
ir/hir/if_node.py — IfNode structured control-flow node.
"""

from typing import Callable, List, Optional, Tuple

from pseudo8051.ir.hir._base import HIRNode, _render_cond, _ann_field, _Cond, _killed_by_seq, _possibly_killed_by_seq, _cond_refs


class IfNode(HIRNode):
    """
    if (condition) { then_nodes } [else { else_nodes }]
    """

    def __init__(self, ea: int, condition: _Cond,
                 then_nodes: List[HIRNode],
                 else_nodes: Optional[List[HIRNode]] = None):
        super().__init__(ea)
        self.condition  = condition
        self.then_nodes = then_nodes
        self.else_nodes = else_nodes or []

    def child_body_groups(self):
        groups = [(0, self.then_nodes)]
        if self.else_nodes:
            groups.append((1, self.else_nodes))   # 1 for "} else {"
        return groups

    def map_bodies(self, fn: Callable[[List[HIRNode]], List[HIRNode]]) -> "IfNode":
        return self.copy_meta_to(IfNode(self.ea, self.condition, fn(self.then_nodes), fn(self.else_nodes)))

    def definitely_killed(self) -> frozenset:
        """Registers killed on ALL paths: intersection of both branch kill sets."""
        return _killed_by_seq(self.then_nodes) & _killed_by_seq(self.else_nodes)

    def possibly_killed(self) -> frozenset:
        """Registers killed on ANY path: union of both branch kill sets."""
        return _possibly_killed_by_seq(self.then_nodes) | _possibly_killed_by_seq(self.else_nodes)

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

    def name_refs(self) -> frozenset:
        body_refs = frozenset().union(*(n.name_refs() for n in self.then_nodes + self.else_nodes))
        return _cond_refs(self.condition) | body_refs

    def map_exprs(self, fn) -> "IfNode":
        new_cond = fn(self.condition)
        if new_cond is self.condition:
            return self
        return IfNode(self.ea, new_cond, self.then_nodes, self.else_nodes)

    def replace_condition(self, new_cond) -> "IfNode":
        return self.copy_meta_to(IfNode(self.ea, new_cond, self.then_nodes, self.else_nodes))

    def ann_lines(self) -> List[str]:
        return (["IfNode"] + _ann_field("cond", self.condition)
                + [f"  then: {len(self.then_nodes)} nodes",
                   f"  else: {len(self.else_nodes)} nodes"])
