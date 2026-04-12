"""
ir/expr.py — Expression tree nodes for the 8051 pseudocode HIR.

Atomic nodes: Regs (via Reg/RegGroup factories), Const, Name
Composite nodes: XRAMRef, IRAMRef, CROMRef, BinOp, UnaryOp, Call, Cast

Regs is the single class for all register expression nodes.
Reg(name) and RegGroup(regs) are factory functions that return Regs instances.

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

class Regs(Expr):
    """Common base for single-register (Reg) and multi-register (RegGroup) expressions.

    Provides a uniform interface with set operations so callers can use
    ``isinstance(e, Regs)`` and access ``e.names``, ``e.is_single``,
    ``e.reg_set()``, etc. without special-casing Reg vs RegGroup.

    names: ordered tuple of register names, high byte first (e.g. ('R6', 'R7')).
    brace: if True render as ``{R6, R7}`` (used for 8051 {B, A} destinations).
    alias: optional display name used only for rendering — does not affect identity.
    """

    __slots__ = ("names", "brace", "alias")

    def __init__(self, names: Tuple[str, ...], brace: bool = False,
                 alias: Optional[str] = None):
        self.names = tuple(names)
        self.brace = brace
        self.alias = alias

    # ── Uniform register-set interface ──

    @property
    def is_single(self) -> bool:
        """True when this expression covers exactly one register."""
        return len(self.names) == 1

    def reg_set(self) -> frozenset:
        """All register names as an unordered frozenset."""
        return frozenset(self.names)

    def overlaps(self, other: "Regs") -> bool:
        """True when self and other share at least one register name."""
        return bool(self.reg_set() & other.reg_set())

    def is_subset_of(self, other: "Regs") -> bool:
        """True when every register in self is also in other."""
        return self.reg_set() <= other.reg_set()

    def __contains__(self, r: str) -> bool:
        return r in self.names

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    # ── Expr interface ──

    def render(self, outer_prec: int = 0) -> str:
        if self.alias:
            return self.alias
        if self.brace:
            return "{" + ", ".join(self.names) + "}"
        if len(self.names) == 1:
            return self.names[0]
        return "".join(self.names)

    @property
    def name(self) -> str:
        """The register name string (backward-compat access to names[0] for single regs)."""
        return self.names[0]

    @property
    def regs(self) -> Tuple[str, ...]:
        """The register names tuple (backward-compat access to self.names)."""
        return self.names

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Regs)
                and self.names == other.names
                and self.brace == other.brace)

    def __hash__(self) -> int:
        return hash(("Regs", self.names, self.brace))

    def __repr__(self) -> str:
        if self.alias:
            return f"Regs({self.names!r}, alias={self.alias!r})"
        if self.brace:
            return f"Regs({self.names!r}, brace=True)"
        return f"Regs({self.names!r})"


def Reg(name: str, alias: Optional[str] = None) -> "Regs":
    """Factory: single-register Regs node. Reg('A') → Regs(('A',))."""
    return Regs((name,), alias=alias)


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


def RegGroup(regs: Tuple[str, ...], brace: bool = False,
             alias: Optional[str] = None) -> "Regs":
    """Factory: multi-register Regs node. RegGroup(('R6','R7')) → Regs(('R6','R7'))."""
    return Regs(regs, brace=brace, alias=alias)


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

    def children(self) -> List["Expr"]:
        return [self.a_arg, self.c_arg]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
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


class ArrayRef(Expr):
    """Array subscript: ArrayRef(Name("foo"), Const(2)) → "foo[2]"."""

    __slots__ = ("base", "index")

    def __init__(self, base: Expr, index: Expr):
        self.base  = base
        self.index = index

    def render(self, outer_prec: int = 0) -> str:
        return f"{self.base.render()}[{self.index.render()}]"

    def children(self) -> List["Expr"]:
        return [self.base, self.index]

    def rebuild(self, new_children: List["Expr"]) -> "Expr":
        return ArrayRef(new_children[0], new_children[1])

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ArrayRef)
                and self.base  == other.base
                and self.index == other.index)

    def __hash__(self) -> int:
        return hash(("ArrayRef", self.base, self.index))

    def __repr__(self) -> str:
        return f"ArrayRef({self.base!r}, {self.index!r})"


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
