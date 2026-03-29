"""
handlers/__init__.py — handler registry + DefaultHandler.

All mnemonic handlers are registered here.  Instruction.lift() / .use() /
.defs() look up the handler via HANDLERS[mnemonic].
"""

from abc import ABC, abstractmethod
from typing import List

import idc

# Import all handler modules so their registration runs at import time.
from .mov        import MovHandler, MovxHandler, MovcHandler, PushHandler, PopHandler, XchHandler, XchdHandler
from .arithmetic import AddHandler, AddcHandler, SubbHandler, IncHandler, DecHandler, MulHandler, DivHandler, DaHandler
from .logic      import AnlHandler, OrlHandler, XrlHandler, ClrHandler, SetbHandler, CplHandler
from .logic      import RlHandler, RlcHandler, RrHandler, RrcHandler, SwapHandler
from .branch     import SjmpHandler, JzHandler, JnzHandler, JcHandler, JncHandler
from .branch     import JbHandler, JnbHandler, JbcHandler, CjneHandler, DjnzHandler
from .branch     import JmpAtADptrHandler
from .call       import LcallHandler, RetHandler, RetiHandler, NopHandler


class DefaultHandler:
    """Fallback for unrecognised mnemonics — emits a raw comment."""

    def use(self, insn) -> frozenset:
        return frozenset()

    def defs(self, insn) -> frozenset:
        return frozenset()

    def lift(self, insn, state=None) -> List[str]:
        raw_op0 = idc.print_operand(insn.ea, 0)
        raw_op1 = idc.print_operand(insn.ea, 1)
        ops_str = f"{raw_op0}, {raw_op1}".strip(", ")
        mnem    = insn.get_canon_mnem().upper()
        return [f"/* {mnem} {ops_str} */"]


_default = DefaultHandler()

# ── Handler registry ──────────────────────────────────────────────────────────
# Maps upper-case mnemonic string → handler instance.

HANDLERS = {
    # Data movement
    "MOV":   MovHandler(),
    "MOVX":  MovxHandler(),
    "MOVC":  MovcHandler(),
    "PUSH":  PushHandler(),
    "POP":   PopHandler(),
    "XCH":   XchHandler(),
    "XCHD":  XchdHandler(),

    # Arithmetic
    "ADD":   AddHandler(),
    "ADDC":  AddcHandler(),
    "SUBB":  SubbHandler(),
    "INC":   IncHandler(),
    "DEC":   DecHandler(),
    "MUL":   MulHandler(),
    "DIV":   DivHandler(),
    "DA":    DaHandler(),

    # Logic
    "ANL":   AnlHandler(),
    "ORL":   OrlHandler(),
    "XRL":   XrlHandler(),
    "CLR":   ClrHandler(),
    "SETB":  SetbHandler(),
    "CPL":   CplHandler(),
    "RL":    RlHandler(),
    "RLC":   RlcHandler(),
    "RR":    RrHandler(),
    "RRC":   RrcHandler(),
    "SWAP":  SwapHandler(),

    # Branches
    "SJMP":  SjmpHandler(),
    "LJMP":  SjmpHandler(),
    "AJMP":  SjmpHandler(),
    "JMP":   JmpAtADptrHandler(),
    "JZ":    JzHandler(),
    "JNZ":   JnzHandler(),
    "JC":    JcHandler(),
    "JNC":   JncHandler(),
    "JB":    JbHandler(),
    "JNB":   JnbHandler(),
    "JBC":   JbcHandler(),
    "CJNE":  CjneHandler(),
    "DJNZ":  DjnzHandler(),

    # Calls / returns
    "LCALL": LcallHandler(),
    "ACALL": LcallHandler(),
    "CALL":  LcallHandler(),
    "RET":   RetHandler(),
    "RETI":  RetiHandler(),
    "NOP":   NopHandler(),
}


def get_handler(mnemonic: str):
    """Return the handler for mnemonic, or DefaultHandler if not found."""
    return HANDLERS.get(mnemonic.upper(), _default)
