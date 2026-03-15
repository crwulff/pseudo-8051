"""
ir/hir.py — High-level IR nodes.

All nodes carry an ea (source address) for viewer double-click navigation.
render() returns a list of (ea, indented_text) tuples.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


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


class Statement(HIRNode):
    """A single C-like statement string (already formatted by a handler)."""

    def __init__(self, ea: int, text: str):
        super().__init__(ea)
        self.text = text

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        return [(self.ea, f"{self._ind(indent)}{self.text}")]


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
        # Labels are not indented by the passed indent; they typically sit at
        # the outermost scope.
        return [
            (self.ea, ""),
            (self.ea, f"{self.name}:"),
        ]


class IfNode(HIRNode):
    """
    if (condition) { then_nodes } [else { else_nodes }]
    """

    def __init__(self, ea: int, condition: str,
                 then_nodes: List[HIRNode],
                 else_nodes: Optional[List[HIRNode]] = None):
        super().__init__(ea)
        self.condition  = condition
        self.then_nodes = then_nodes
        self.else_nodes = else_nodes or []

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}if ({self.condition}) {{"))
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
    """

    def __init__(self, ea: int, condition: str,
                 body_nodes: List[HIRNode]):
        super().__init__(ea)
        self.condition  = condition
        self.body_nodes = body_nodes

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}while ({self.condition}) {{"))
        for node in self.body_nodes:
            lines.extend(node.render(indent + 1))
        lines.append((self.ea, f"{ind}}}"))
        return lines


class ForNode(HIRNode):
    """
    for (init; condition; update) { body_nodes }
    """

    def __init__(self, ea: int, init: str, condition: str, update: str,
                 body_nodes: List[HIRNode]):
        super().__init__(ea)
        self.init       = init
        self.condition  = condition
        self.update     = update
        self.body_nodes = body_nodes

    def render(self, indent: int = 0) -> List[Tuple[int, str]]:
        ind = self._ind(indent)
        lines: List[Tuple[int, str]] = []
        lines.append((self.ea, f"{ind}for ({self.init}; {self.condition}; {self.update}) {{"))
        for node in self.body_nodes:
            lines.extend(node.render(indent + 1))
        lines.append((self.ea, f"{ind}}}"))
        return lines
