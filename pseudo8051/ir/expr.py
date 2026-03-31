"""
ir/expr.py — Expression tree nodes for the 8051 pseudocode HIR.

Atomic nodes: Reg, Const, Name
Composite nodes: XRAMRef, IRAMRef, CROMRef, RegGroup, BinOp, UnaryOp, Call, Cast

Each node implements:
  render(outer_prec=99) -> str  — produce C-like text, inserting parens as needed
  children() -> List[Expr]      — direct child expressions (for tree walking)
  rebuild(new_children)         — return copy with replaced children
  __eq__ / __hash__             — needed for use in replacement dicts and sets
"""

import sys
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


# ── Operator precedence (lower number = tighter binding) ─────────────────────
# Unary = 1 (tightest)
# */%   = 2
# +-    = 3
# <<>>  = 4
# <><=>=  = 5
# ==!=  = 6
# &     = 7
# ^     = 8
# |     = 9
# &&    = 10
# ||    = 11

_BIN_PREC = {
    '*': 2, '/': 2, '%': 2,
    '+': 3, '-': 3,
    '<<': 4, '>>': 4,
    '<': 5, '>': 5, '<=': 5, '>=': 5,
    '==': 6, '!=': 6,
    '&': 7,
    '^': 8,
    '|': 9,
    '&&': 10,
    '||': 11,
}
_UNARY_PREC = 1  # tighter than any binary op


def _const_str(value: int) -> str:
    """Format an integer constant matching Operand.render() behaviour."""
    _c = sys.modules.get("pseudo8051.constants")
    use_hex = getattr(_c, "USE_HEX", True) if _c else True
    return str(value) if (not use_hex or value <= 9) else hex(value)


# ── Abstract base ─────────────────────────────────────────────────────────────

class Expr(ABC):
    """Base class for all expression-tree nodes."""

    @abstractmethod
    def render(self, outer_prec: int = 0) -> str:
        """
        Return a C-like text representation.

        outer_prec is the precedence level of the *enclosing* expression.
        Composite nodes wrap themselves in parens when their own precedence is
        weaker (higher number) than outer_prec.
        """
        ...

    def children(self) -> List["Expr"]:
        """Return direct child Expr nodes.  Leaf nodes return []."""
        return []

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        """Return a copy of this node with children replaced.
        Must accept exactly len(self.children()) elements."""
        return self  # leaf: no children, return unchanged

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.render()})"


# ── Atomic nodes ──────────────────────────────────────────────────────────────

class Reg(Expr):
    """A named register: Reg("A"), Reg("R7"), Reg("C"), Reg("DPTR")."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def render(self, outer_prec: int = 0) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Reg) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("Reg", self.name))

    def __repr__(self) -> str:
        return f"Reg({self.name!r})"


class Const(Expr):
    """An integer constant: Const(0x5d), Const(0)."""

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = value

    def render(self, outer_prec: int = 0) -> str:
        return _const_str(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Const) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("Const", self.value))

    def __repr__(self) -> str:
        return f"Const({self.value!r})"


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


# ── Composite nodes ───────────────────────────────────────────────────────────

class XRAMRef(Expr):
    """External RAM access: XRAMRef(Name("EXT_DC8A"))  → "XRAM[EXT_DC8A]"."""

    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"XRAM[{self.inner.render()}]"

    def children(self) -> List["Expr"]:
        return [self.inner]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return XRAMRef(new_children[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, XRAMRef) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("XRAMRef", self.inner))

    def __repr__(self) -> str:
        return f"XRAMRef({self.inner!r})"


class IRAMRef(Expr):
    """Internal RAM indirect access: IRAMRef(Reg("R0")) → "IRAM[R0]"."""

    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"IRAM[{self.inner.render()}]"

    def children(self) -> List["Expr"]:
        return [self.inner]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return IRAMRef(new_children[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IRAMRef) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("IRAMRef", self.inner))

    def __repr__(self) -> str:
        return f"IRAMRef({self.inner!r})"


class CROMRef(Expr):
    """Code ROM indirect access: CROMRef(BinOp(...)) → "CROM[A + DPTR]"."""

    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"CROM[{self.inner.render()}]"

    def children(self) -> List["Expr"]:
        return [self.inner]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return CROMRef(new_children[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CROMRef) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("CROMRef", self.inner))

    def __repr__(self) -> str:
        return f"CROMRef({self.inner!r})"


class RegGroup(Expr):
    """
    A multi-register group.

    RegGroup(("R6", "R7"))              → "R6R7"
    RegGroup(("B", "A"), brace=True)    → "{B, A}"
    """

    __slots__ = ("regs", "brace")

    def __init__(self, regs: Tuple[str, ...], brace: bool = False):
        self.regs  = tuple(regs)
        self.brace = brace

    def render(self, outer_prec: int = 0) -> str:
        if self.brace:
            return "{" + ", ".join(self.regs) + "}"
        return "".join(self.regs)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, RegGroup)
                and self.regs == other.regs
                and self.brace == other.brace)

    def __hash__(self) -> int:
        return hash(("RegGroup", self.regs, self.brace))

    def __repr__(self) -> str:
        if self.brace:
            return f"RegGroup({self.regs!r}, brace=True)"
        return f"RegGroup({self.regs!r})"


class BinOp(Expr):
    """
    Binary operation: BinOp(Reg("A"), "+", Const(5)) → "A + 5".

    render() wraps in parens when outer_prec < this node's prec (the
    enclosing context binds more tightly).
    """

    __slots__ = ("lhs", "op", "rhs")

    def __init__(self, lhs: Expr, op: str, rhs: Expr):
        self.lhs = lhs
        self.op  = op
        self.rhs = rhs

    @property
    def prec(self) -> int:
        return _BIN_PREC.get(self.op, 99)

    def render(self, outer_prec: int = 99) -> str:
        my_prec = self.prec
        # Add parens when outer context binds more tightly (lower prec number).
        # Default outer_prec=99 (top-level / no parent) → no parens added.
        need_parens = outer_prec < my_prec
        inner = f"{self.lhs.render(my_prec)} {self.op} {self.rhs.render(my_prec)}"
        return f"({inner})" if need_parens else inner

    def children(self) -> List["Expr"]:
        return [self.lhs, self.rhs]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return BinOp(new_children[0], self.op, new_children[1])

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, BinOp)
                and self.lhs == other.lhs
                and self.op  == other.op
                and self.rhs == other.rhs)

    def __hash__(self) -> int:
        return hash(("BinOp", self.lhs, self.op, self.rhs))

    def __repr__(self) -> str:
        return f"BinOp({self.lhs!r}, {self.op!r}, {self.rhs!r})"


class UnaryOp(Expr):
    """
    Unary operation.

    UnaryOp("--", Reg("R7"), post=False)  → "--R7"   (pre-decrement)
    UnaryOp("++", Reg("R7"), post=True)   → "R7++"   (post-increment)
    UnaryOp("!",  Reg("C"))               → "!C"
    UnaryOp("~",  Reg("A"))               → "~A"
    """

    __slots__ = ("op", "operand", "post")

    def __init__(self, op: str, operand: Expr, post: bool = False):
        self.op      = op
        self.operand = operand
        self.post    = post

    def render(self, outer_prec: int = 0) -> str:
        # Unary has tightest binding; parens only needed when outer is tighter
        # (which doesn't normally happen for standard C precedence).
        inner = self.operand.render(_UNARY_PREC)
        if self.post:
            return f"{inner}{self.op}"
        return f"{self.op}{inner}"

    def children(self) -> List["Expr"]:
        return [self.operand]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return UnaryOp(self.op, new_children[0], self.post)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, UnaryOp)
                and self.op      == other.op
                and self.operand == other.operand
                and self.post    == other.post)

    def __hash__(self) -> int:
        return hash(("UnaryOp", self.op, self.operand, self.post))

    def __repr__(self) -> str:
        return f"UnaryOp({self.op!r}, {self.operand!r}, post={self.post})"


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

    def children(self) -> List["Expr"]:
        return list(self.args)

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return Call(self.func_name, new_children)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Call)
                and self.func_name == other.func_name
                and self.args      == other.args)

    def __hash__(self) -> int:
        return hash(("Call", self.func_name, tuple(self.args)))

    def __repr__(self) -> str:
        return f"Call({self.func_name!r}, {self.args!r})"


class Paren(Expr):
    """Explicit parenthesis wrapper — always renders as (inner)."""

    __slots__ = ("inner",)

    def __init__(self, inner: Expr):
        self.inner = inner

    def render(self, outer_prec: int = 0) -> str:
        return f"({self.inner.render()})"

    def children(self) -> List["Expr"]:
        return [self.inner]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return Paren(new_children[0])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Paren) and self.inner == other.inner

    def __hash__(self) -> int:
        return hash(("Paren", self.inner))

    def __repr__(self) -> str:
        return f"Paren({self.inner!r})"


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

    def children(self) -> List["Expr"]:
        return [self.inner]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return Cast(self.type_str, new_children[0])

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Cast)
                and self.type_str == other.type_str
                and self.inner    == other.inner)

    def __hash__(self) -> int:
        return hash(("Cast", self.type_str, self.inner))

    def __repr__(self) -> str:
        return f"Cast({self.type_str!r}, {self.inner!r})"
