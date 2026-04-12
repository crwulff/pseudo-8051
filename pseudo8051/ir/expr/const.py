"""
ir/expr/const.py — Const: integer constant expression node.
"""

from typing import Optional

from pseudo8051.ir.expr._base import Expr
from pseudo8051.ir.expr._prec import _const_str


class Const(Expr):
    """An integer constant: Const(0x5d), Const(0).

    alias: optional display name (e.g. an enum member or type-padded hex string)
    used only for rendering.  Integer identity (eq/hash/comparisons) always uses
    self.value.
    """

    __slots__ = ("value", "alias")

    def __init__(self, value: int, alias: Optional[str] = None):
        self.value = value
        self.alias = alias

    def render(self, outer_prec: int = 0) -> str:
        return self.alias if self.alias else _const_str(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Const) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("Const", self.value))

    def __repr__(self) -> str:
        if self.alias:
            return f"Const({self.value!r}, alias={self.alias!r})"
        return f"Const({self.value!r})"
