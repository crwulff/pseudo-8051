"""
ir/expr/array_ref.py — ArrayRef: array subscript expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr


class ArrayRef(Expr):
    """Array subscript: ArrayRef(Name("foo"), Const(2)) → "foo[2]"."""

    __slots__ = ("base", "index")

    def __init__(self, base: Expr, index: Expr):
        self.base  = base
        self.index = index

    def render(self, outer_prec: int = 0) -> str:
        return f"{self.base.render()}[{self.index.render()}]"

    def children(self) -> List[Expr]:
        return [self.base, self.index]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return ArrayRef(new_children[0], new_children[1])

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ArrayRef)
                and self.base  == other.base
                and self.index == other.index)

    def __hash__(self) -> int:
        return hash(("ArrayRef", self.base, self.index))

    def __repr__(self) -> str:
        return f"ArrayRef({self.base!r}, {self.index!r})"
