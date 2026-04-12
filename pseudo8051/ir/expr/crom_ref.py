"""
ir/expr/crom_ref.py — CROMRef: code ROM indirect access expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr


class CROMRef(Expr):
    """Code ROM indirect access: CROMRef(BinOp(...)) → "CROM[A + DPTR]"."""

    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"CROM[{self.inner.render()}]"

    def children(self) -> List[Expr]:
        return [self.inner]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return CROMRef(new_children[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CROMRef) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("CROMRef", self.inner))

    def __repr__(self) -> str:
        return f"CROMRef({self.inner!r})"
