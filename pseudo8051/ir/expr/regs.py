"""
ir/expr/regs.py — Regs, Reg, RegGroup: register expression nodes.
"""

from typing import Optional, Tuple

from pseudo8051.ir.expr._base import Expr


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


def RegGroup(regs: Tuple[str, ...], brace: bool = False,
             alias: Optional[str] = None) -> "Regs":
    """Factory: multi-register Regs node. RegGroup(('R6','R7')) → Regs(('R6','R7'))."""
    return Regs(regs, brace=brace, alias=alias)
