"""
ir/expr/iram_ref.py — IRAMRef: internal RAM indirect access expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr


class IRAMRef(Expr):
    """Internal RAM indirect access: IRAMRef(Reg("R0")) → "IRAM[R0]"."""

    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"IRAM[{self.inner.render()}]"

    def children(self) -> List[Expr]:
        return [self.inner]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return IRAMRef(new_children[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IRAMRef) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("IRAMRef", self.inner))

    def __repr__(self) -> str:
        return f"IRAMRef({self.inner!r})"
