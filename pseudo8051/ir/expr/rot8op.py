"""
ir/expr/rot8op.py — Rot8Op: 8-bit rotate expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr


class Rot8Op(Expr):
    """
    8-bit rotate expression: rol8(A) or ror8(A).

    Represents the 8051 RL / RR instructions as a pure expression node
    so that it renders cleanly and is recognised by RolSwitchPattern.
    """

    __slots__ = ("func_name", "a_arg")

    def __init__(self, func_name: str, a_arg: Expr):
        self.func_name = func_name
        self.a_arg     = a_arg

    def render(self, outer_prec: int = 0) -> str:
        return f"{self.func_name}({self.a_arg.render()})"

    def children(self) -> List[Expr]:
        return [self.a_arg]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return Rot8Op(self.func_name, new_children[0])

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Rot8Op)
                and self.func_name == other.func_name
                and self.a_arg     == other.a_arg)

    def __hash__(self) -> int:
        return hash(("Rot8Op", self.func_name, self.a_arg))

    def __repr__(self) -> str:
        return f"Rot8Op({self.func_name!r}, {self.a_arg!r})"
