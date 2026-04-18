"""
ir/operand.py — Operand class wrapping one IDA operand slot.

render() produces C-like text, migrated from lift_operand() in the old script.
"""

import sys
from typing import Optional

import ida_ua
import ida_name
import idc

from pseudo8051.constants import (
    SFR_NAMES, REG_DPTR,
    PHRASE_AT_R0, PHRASE_AT_R1, PHRASE_AT_DPTR, PHRASE_AT_A_DPTR,
    PARAM_REGS, resolve_ext_addr,
)
from pseudo8051.ir.expr import (
    Expr, Reg, Const, Name, XRAMRef, IRAMRef, CROMRef, BinOp,
)
from pseudo8051.ir.cpstate import CPState


class Operand:
    """Wraps one IDA operand (index n of insn_t) and can render itself."""

    __slots__ = ("_insn", "_n")

    def __init__(self, insn, n: int):
        self._insn = insn
        self._n    = n

    @property
    def type(self) -> int:
        return self._insn.ops[self._n].type

    @property
    def _op(self):
        return self._insn.ops[self._n]

    def render(self, cp_state: Optional[CPState] = None) -> str:
        """
        Return a C-like string for this operand.
        cp_state may supply a known DPTR value for MOVX resolution.
        """
        insn = self._insn
        n    = self._n
        op   = self._op

        if op.type == ida_ua.o_void:
            return ""

        if op.type == ida_ua.o_reg:
            return idc.print_operand(insn.ea, n)

        if op.type == ida_ua.o_imm:
            v = op.value
            _c = sys.modules.get("pseudo8051.constants")
            use_hex = getattr(_c, "USE_HEX", True) if _c else True
            return str(v) if (not use_hex or v <= 9) else hex(v)

        if op.type == ida_ua.o_mem:
            addr = op.addr & 0xFF
            sfr  = SFR_NAMES.get(addr)
            if sfr:
                return sfr
            if addr <= 7:
                return f"R{addr}"
            iname = ida_name.get_name(op.addr)
            if iname:
                return iname
            return f"MEM[{hex(addr)}]"

        if op.type == ida_ua.o_phrase:
            phrase = op.phrase
            if phrase == PHRASE_AT_R0:     return "IRAM[R0]"
            if phrase == PHRASE_AT_R1:     return "IRAM[R1]"
            if phrase == PHRASE_AT_DPTR:
                if cp_state is not None:
                    dptr_val = cp_state.get("DPTR")
                    if dptr_val is not None:
                        return f"XRAM[{resolve_ext_addr(dptr_val)}]"
                return "XRAM[DPTR]"
            if phrase == PHRASE_AT_A_DPTR: return "CROM[A + DPTR]"
            return idc.print_operand(insn.ea, n)

        if op.type in (ida_ua.o_near, ida_ua.o_far):
            page_base = insn.ea & ~0xFFFF
            target_ea = page_base | (op.addr & 0xFFFF)
            name = ida_name.get_name(target_ea)
            return name if name else hex(op.addr & 0xFFFF)

        # o_bit, o_displ, o_idpspec*, … — IDA fallback
        return idc.print_operand(insn.ea, n)

    def registers(self) -> frozenset:
        """
        Return the set of tracked register names this operand reads.
        Used by handlers when building use() sets without duplicating
        operand-decoding logic.
        """
        op = self._op
        insn = self._insn
        n    = self._n

        if op.type == ida_ua.o_reg:
            name = idc.print_operand(insn.ea, n)
            return frozenset({name}) if name in PARAM_REGS else frozenset()

        if op.type == ida_ua.o_phrase:
            phrase = op.phrase
            if phrase == PHRASE_AT_R0:                            return frozenset({"R0"})
            if phrase == PHRASE_AT_R1:                            return frozenset({"R1"})
            if phrase in (PHRASE_AT_DPTR, PHRASE_AT_A_DPTR):     return frozenset({"DPTR"})

        return frozenset()

    def to_expr(self, cp_state: Optional[CPState] = None) -> Expr:
        """
        Return an Expr node for this operand.
        Mirrors render() but returns structured Expr objects instead of strings.
        """
        insn = self._insn
        n    = self._n
        op   = self._op

        if op.type == ida_ua.o_void:
            return Name("")

        if op.type == ida_ua.o_reg:
            return Reg(idc.print_operand(insn.ea, n))

        if op.type == ida_ua.o_imm:
            return Const(op.value)

        if op.type == ida_ua.o_mem:
            addr = op.addr & 0xFF
            sfr  = SFR_NAMES.get(addr)
            if sfr:
                return Reg(sfr)
            if addr <= 7:
                return Reg(f"R{addr}")
            iname = ida_name.get_name(op.addr) or None
            return IRAMRef(Const(addr, alias=iname))

        if op.type == ida_ua.o_phrase:
            phrase = op.phrase
            if phrase == PHRASE_AT_R0:
                return IRAMRef(Reg("R0"))
            if phrase == PHRASE_AT_R1:
                return IRAMRef(Reg("R1"))
            if phrase == PHRASE_AT_DPTR:
                if cp_state is not None:
                    dptr_val = cp_state.get("DPTR")
                    if dptr_val is not None:
                        sym = resolve_ext_addr(dptr_val)
                        alias = sym if sym != hex(dptr_val) else None
                        return XRAMRef(Const(dptr_val, alias=alias))
                return XRAMRef(Reg("DPTR"))
            if phrase == PHRASE_AT_A_DPTR:
                return CROMRef(BinOp(Reg("A"), "+", Reg("DPTR")))
            return Name(idc.print_operand(insn.ea, n))

        if op.type in (ida_ua.o_near, ida_ua.o_far):
            page_base = insn.ea & ~0xFFFF
            target_ea = page_base | (op.addr & 0xFFFF)
            name = ida_name.get_name(target_ea)
            return Name(name) if name else Name(hex(op.addr & 0xFFFF))

        # o_bit, o_displ, o_idpspec*, … — IDA fallback
        return Name(idc.print_operand(insn.ea, n))

    def reg_name(self) -> Optional[str]:
        """
        If this operand is o_reg and the register is in PARAM_REGS, return
        the name; otherwise None.
        """
        op = self._op
        if op.type != ida_ua.o_reg:
            return None
        name = idc.print_operand(self._insn.ea, self._n)
        return name if name in PARAM_REGS else None
