"""
ir/expr/_base.py — Abstract base class for all expression-tree nodes.
"""

from abc import ABC, abstractmethod
from typing import List


class Expr(ABC):
    """Base class for all expression-tree nodes."""

    @abstractmethod
    def render(self, outer_prec: int = 0) -> str:
        """
        Return a C-like text representation.

        outer_prec is the precedence level of the *enclosing* expression.
        Composite nodes wrap themselves in parens when their own precedence is
        weaker (higher number) than outer_prec.
        """
        ...

    def children(self) -> List["Expr"]:
        """Return direct child Expr nodes.  Leaf nodes return []."""
        return []

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        """Return a copy of this node with children replaced.
        Must accept exactly len(self.children()) elements."""
        return self  # leaf: no children, return unchanged

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.render()})"
