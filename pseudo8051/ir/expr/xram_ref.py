"""
ir/expr/xram_ref.py — XRAMRef: external RAM access expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr


class XRAMRef(Expr):
    """External RAM access: XRAMRef(Name("EXT_DC8A"))  → "XRAM[EXT_DC8A]"."""

    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"XRAM[{self.inner.render()}]"

    def children(self) -> List[Expr]:
        return [self.inner]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return XRAMRef(new_children[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, XRAMRef) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("XRAMRef", self.inner))

    def __repr__(self) -> str:
        return f"XRAMRef({self.inner!r})"
