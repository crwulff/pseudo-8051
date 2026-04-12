"""
ir/expr/name.py — Name: symbolic name expression node.
"""

from pseudo8051.ir.expr._base import Expr


class Name(Expr):
    """A symbolic name: Name("EXT_DC8A"), Name("func_name"), Name("H")."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def render(self, outer_prec: int = 0) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Name) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("Name", self.name))

    def __repr__(self) -> str:
        return f"Name({self.name!r})"
