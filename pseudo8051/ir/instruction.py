"""
ir/instruction.py — Instruction wrapper + MnemonicHandler ABC.

Instruction wraps an IDA EA, decodes on demand, and delegates use/defs/lift
to the appropriate MnemonicHandler looked up from handlers/__init__.py.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

import ida_ua
import ida_bytes
import idc

from pseudo8051.ir.operand import Operand

if TYPE_CHECKING:
    from pseudo8051.analysis.constprop import CPState


# ── MnemonicHandler ABC ───────────────────────────────────────────────────────

class MnemonicHandler(ABC):
    """
    One instance per mnemonic family, registered in handlers/__init__.py.
    Owns the use / def / lift / propagate_cp logic for its mnemonic(s).
    """

    @abstractmethod
    def use(self, insn) -> frozenset:
        """Registers whose value BEFORE this instruction may be consumed."""
        ...

    @abstractmethod
    def defs(self, insn) -> frozenset:
        """Registers this instruction unconditionally overwrites."""
        ...

    @abstractmethod
    def lift(self, insn, state: Optional["CPState"] = None) -> List[str]:
        """Return a list of C-like statement strings for this instruction."""
        ...


# ── Instruction ───────────────────────────────────────────────────────────────

class Instruction:
    """Wraps a single decoded 8051 instruction at a given EA."""

    __slots__ = ("ea", "_insn", "_operands")

    def __init__(self, ea: int):
        self.ea        = ea
        self._insn     = None
        self._operands = None

    # ── Decoding ──────────────────────────────────────────────────────────

    def decode(self):
        """Decode and cache the insn_t; return it."""
        if self._insn is None:
            insn = ida_ua.insn_t()
            size = ida_ua.decode_insn(insn, self.ea)
            if size > 0:
                self._insn = insn
        return self._insn

    @property
    def insn(self):
        return self.decode()

    @property
    def size(self) -> int:
        insn = self.insn
        return insn.size if insn else 1

    @property
    def mnemonic(self) -> str:
        insn = self.insn
        return insn.get_canon_mnem().upper() if insn else ""

    @property
    def operands(self) -> List[Operand]:
        if self._operands is None:
            insn = self.insn
            if insn:
                self._operands = [Operand(insn, i) for i in range(6)
                                  if insn.ops[i].type != ida_ua.o_void]
            else:
                self._operands = []
        return self._operands

    # ── Handler delegation ────────────────────────────────────────────────

    def _handler(self):
        from pseudo8051.handlers import get_handler
        return get_handler(self.mnemonic)

    def use(self) -> frozenset:
        insn = self.insn
        return self._handler().use(insn) if insn else frozenset()

    def defs(self) -> frozenset:
        insn = self.insn
        return self._handler().defs(insn) if insn else frozenset()

    def lift(self, state: Optional["CPState"] = None) -> List[str]:
        insn = self.insn
        return self._handler().lift(insn, state) if insn else []

    # ── Branch classification ─────────────────────────────────────────────

    _BRANCH_MNEMS = frozenset({
        "SJMP", "LJMP", "AJMP",
        "JZ", "JNZ", "JC", "JNC", "JB", "JNB", "JBC", "CJNE", "DJNZ",
    })
    _COND_BRANCH_MNEMS = frozenset({
        "JZ", "JNZ", "JC", "JNC", "JB", "JNB", "JBC", "CJNE", "DJNZ",
    })
    _CALL_MNEMS = frozenset({"LCALL", "ACALL", "CALL"})
    _RET_MNEMS  = frozenset({"RET", "RETI"})

    def is_branch(self) -> bool:
        return self.mnemonic in self._BRANCH_MNEMS

    def is_unconditional_branch(self) -> bool:
        return self.mnemonic in {"SJMP", "LJMP", "AJMP"}

    def is_conditional_branch(self) -> bool:
        return self.mnemonic in self._COND_BRANCH_MNEMS

    def is_call(self) -> bool:
        return self.mnemonic in self._CALL_MNEMS

    def is_return(self) -> bool:
        return self.mnemonic in self._RET_MNEMS

    def branch_targets(self) -> List[int]:
        """
        Return explicit branch-target EAs from near/far operands.
        Does NOT include the fall-through successor.
        """
        insn = self.insn
        if not insn:
            return []
        targets = []
        page_base = insn.ea & ~0xFFFF
        for i in range(6):
            op = insn.ops[i]
            if op.type in (ida_ua.o_near, ida_ua.o_far):
                targets.append(page_base | (op.addr & 0xFFFF))
        return targets

    def __repr__(self) -> str:
        return f"<Instruction ea={hex(self.ea)} mnem={self.mnemonic!r}>"
