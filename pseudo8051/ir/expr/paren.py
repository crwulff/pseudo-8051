"""
ir/expr/paren.py — Paren: explicit parenthesis wrapper expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr


class Paren(Expr):
    """Explicit parenthesis wrapper — always renders as (inner)."""

    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"({self.inner.render()})"

    def children(self) -> List[Expr]:
        return [self.inner]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return Paren(new_children[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Paren) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("Paren", self.inner))

    def __repr__(self) -> str:
        return f"Paren({self.inner!r})"
