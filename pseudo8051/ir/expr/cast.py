"""
ir/expr/cast.py — Cast: type cast expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr
from pseudo8051.ir.expr._prec import _UNARY_PREC


class Cast(Expr):
    """
    Type cast: Cast("uint8_t", Reg("A")) → "(uint8_t)A".
    """

    __slots__ = ("type_str", "inner")

    def __init__(self, type_str: str, inner: Expr):
        self.type_str = type_str
        self.inner    = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"({self.type_str}){self.inner.render(_UNARY_PREC)}"

    def children(self) -> List[Expr]:
        return [self.inner]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return Cast(self.type_str, new_children[0])

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Cast)
                and self.type_str == other.type_str
                and self.inner    == other.inner)

    def __hash__(self) -> int:
        return hash(("Cast", self.type_str, self.inner))

    def __repr__(self) -> str:
        return f"Cast({self.type_str!r}, {self.inner!r})"
