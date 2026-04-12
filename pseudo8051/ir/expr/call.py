"""
ir/expr/call.py — Call: function call expression node.
"""

from typing import List

from pseudo8051.ir.expr._base import Expr


class Call(Expr):
    """
    Function call expression: Call("func", [Reg("R7"), ...]) → "func(R7, ...)".
    """

    __slots__ = ("func_name", "args")

    def __init__(self, func_name: str, args: List[Expr]):
        self.func_name = func_name
        self.args      = list(args)

    def render(self, outer_prec: int = 0) -> str:
        args_str = ", ".join(a.render() for a in self.args)
        return f"{self.func_name}({args_str})"

    def children(self) -> List[Expr]:
        return list(self.args)

    def rebuild(self, new_children: List[Expr]) -> Expr:
        return Call(self.func_name, new_children)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Call)
                and self.func_name == other.func_name
                and self.args      == other.args)

    def __hash__(self) -> int:
        return hash(("Call", self.func_name, tuple(self.args)))

    def __repr__(self) -> str:
        return f"Call({self.func_name!r}, {self.args!r})"
