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

    def _direct_side_effect_regs(self) -> frozenset:
        """Register names written as a direct side effect of THIS node only.

        Override in subclasses that have side effects (e.g. UnaryOp ++/--).
        Does NOT recurse into children; side_effect_regs() does that.
        """
        return frozenset()

    def side_effect_regs(self) -> frozenset:
        """Register names written as a side effect of evaluating this expression
        or any of its sub-expressions (recursive).

        Increment/decrement operations on a register operand are the primary
        source of side effects; e.g. XRAMRef(UnaryOp('++', DPTR)) reports DPTR.
        """
        result = self._direct_side_effect_regs()
        for child in self.children():
            result = result | child.side_effect_regs()
        return result

    def children(self) -> List["Expr"]:
        """Return direct child Expr nodes.  Leaf nodes return []."""
        return []

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        """Return a copy of this node with children replaced.
        Must accept exactly len(self.children()) elements."""
        return self  # leaf: no children, return unchanged

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.render()})"
