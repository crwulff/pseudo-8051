"""
ir/expr/rot9op.py — Rot9Op: 9-bit rotate-through-carry expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr


class Rot9Op(Expr):
    """
    9-bit rotate-through-carry expression: rol9(A, C) or ror9(A, C).

    Represents the 8051 RLC / RRC instructions as a pure expression node
    (not a Call) so that it does not kill type-group annotations in the
    AnnotationPass.  The result is the new value of A; the carry update
    is recorded as a side-effect via RlcHandler/RrcHandler.defs().
    """

    __slots__ = ("func_name", "a_arg", "c_arg")

    def __init__(self, func_name: str, a_arg: Expr, c_arg: Expr):
        self.func_name = func_name
        self.a_arg     = a_arg
        self.c_arg     = c_arg

    def render(self, outer_prec: int = 0) -> str:
        return f"{self.func_name}({self.a_arg.render()}, {self.c_arg.render()})"

    def children(self) -> List[Expr]:
        return [self.a_arg, self.c_arg]

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return Rot9Op(self.func_name, new_children[0], new_children[1])

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Rot9Op)
                and self.func_name == other.func_name
                and self.a_arg     == other.a_arg
                and self.c_arg     == other.c_arg)

    def __hash__(self) -> int:
        return hash(("Rot9Op", self.func_name, self.a_arg, self.c_arg))

    def __repr__(self) -> str:
        return f"Rot9Op({self.func_name!r}, {self.a_arg!r}, {self.c_arg!r})"
