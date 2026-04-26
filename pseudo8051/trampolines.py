"""
trampolines.py — Cross-page trampoline detection and resolution.

A cross-page trampoline is a 2-instruction function:
    MOV  DPTR, #target_func_addr
    LJMP code_X_call_page_Y

where the LJMP target (directly or via one hop) is a page-switch helper that
writes a constant page number to the page-bank register via MOVX @DPTR, A.

Exports:
    get_trampoline_target(func_ea) -> Optional[int]
    resolve_callee(callee_name)    -> str
    clear_cache()
"""

import traceback
from typing import Dict, Optional

from pseudo8051.constants import REG_DPTR, PHRASE_AT_DPTR, dbg

# func_ea → real target EA, or None if not a trampoline.
_cache: Dict[int, Optional[int]] = {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_page_in_func(func_ea: int, depth: int = 0) -> Optional[int]:
    """
    Scan the function at func_ea for the page-switch pattern:
        MOV  A, #N          (1 ≤ N ≤ 15)
        …(within 5 insns)…
        MOVX @DPTR, A

    If not found and depth < 2, follow a terminal unconditional jump to a
    different function entry and recurse.

    Returns the page number N, or None.
    """
    try:
        import idc, idautils, ida_ua, ida_funcs, ida_name

        heads = list(idautils.FuncItems(func_ea))
        page_candidate: Optional[int] = None
        page_insn_idx: int = -1

        for idx, ea in enumerate(heads):
            insn = ida_ua.insn_t()
            if not ida_ua.decode_insn(insn, ea):
                continue
            mnem = idc.print_insn_mnem(ea).upper()

            # MOV A, #N — remember candidate page number
            if (mnem == "MOV"
                    and insn.ops[0].type == ida_ua.o_reg
                    and idc.print_operand(ea, 0) == "A"
                    and insn.ops[1].type == ida_ua.o_imm
                    and 1 <= (insn.ops[1].value & 0xFF) <= 15):
                page_candidate = insn.ops[1].value & 0xFF
                page_insn_idx = idx
                continue

            # MOVX @DPTR, A — confirm page write
            if (mnem == "MOVX"
                    and insn.ops[0].type == ida_ua.o_phrase
                    and insn.ops[0].phrase == PHRASE_AT_DPTR
                    and idc.print_operand(ea, 1) == "A"
                    and page_candidate is not None
                    and idx - page_insn_idx <= 5):
                return page_candidate

            # Any other write to A invalidates the candidate
            if page_candidate is not None and idx > page_insn_idx:
                if (mnem in ("MOV", "MOVX", "MOVC", "ADD", "ADDC", "SUBB",
                             "ANL", "ORL", "XRL", "CLR", "CPL", "RL", "RLC",
                             "RR", "RRC", "SWAP", "DA", "XCH", "XCHD", "INC", "DEC")
                        and insn.ops[0].type == ida_ua.o_reg
                        and idc.print_operand(ea, 0) == "A"):
                    page_candidate = None
                    page_insn_idx = -1

        # Not found directly — follow a terminal unconditional jump (one hop)
        if depth < 2 and heads:
            last_ea = heads[-1]
            insn = ida_ua.insn_t()
            if ida_ua.decode_insn(insn, last_ea):
                mnem = idc.print_insn_mnem(last_ea).upper()
                if mnem in ("LJMP", "SJMP", "AJMP"):
                    op = insn.ops[0]
                    if op.type in (ida_ua.o_near, ida_ua.o_far):
                        page_base = func_ea & ~0xFFFF
                        target_ea = page_base | (op.addr & 0xFFFF)
                        target_fn = ida_funcs.get_func(target_ea)
                        if (target_fn is not None
                                and target_fn.start_ea == target_ea
                                and target_fn.start_ea != func_ea):
                            dbg("trampolines", f"  _detect_page: {hex(func_ea)} depth={depth} following LJMP → {hex(target_ea)}")
                            return _detect_page_in_func(target_ea, depth + 1)
                        else:
                            dbg("trampolines", f"  _detect_page: {hex(func_ea)} LJMP target {hex(target_ea)} is not a different function entry")
                else:
                    dbg("trampolines", f"  _detect_page: {hex(func_ea)} last insn is {mnem}, not a jump")
        dbg("trampolines", f"  _detect_page: {hex(func_ea)} depth={depth} → None (no pattern found, {len(heads)} insns)")
        return None
    except Exception:
        dbg("trampolines", f"  _detect_page: {hex(func_ea)} exception: {traceback.format_exc()}")
        return None


def _page_segment_base(page_num: int) -> int:
    """
    Return the IDA segment base EA for code page page_num.
    Scans segments for one whose name ends with the page number string
    (e.g. "code_6"), falling back to page_num * 0x10000.
    """
    try:
        import idautils, idc
        suffix = str(page_num)
        for seg_ea in idautils.Segments():
            name = idc.get_segm_name(seg_ea) or ""
            # Match "code_6", "code_6_xxx", etc. — name must end with _<N> or be <N>
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1] == suffix:
                return seg_ea & ~0xFFFF
            if name == suffix:
                return seg_ea & ~0xFFFF
    except Exception:
        pass
    return page_num * 0x10000


def _detect_trampoline(func_ea: int) -> Optional[int]:
    """
    Core detection: return real target EA if func_ea is a cross-page trampoline.
    """
    try:
        import idc, ida_ua, ida_funcs, ida_name

        BADADDR = idc.BADADDR

        # Decode the first two instructions at func_ea directly.
        # Do NOT use FuncItems — IDA often groups multiple adjacent 2-insn trampolines
        # into one "function", so len(FuncItems) != 2 even for valid trampolines.
        insn0 = ida_ua.insn_t()
        if not ida_ua.decode_insn(insn0, func_ea):
            dbg("trampolines", f"_detect_trampoline {hex(func_ea)}: failed to decode insn0")
            return None
        insn1 = ida_ua.insn_t()
        next_ea = func_ea + insn0.size
        if not ida_ua.decode_insn(insn1, next_ea):
            dbg("trampolines", f"_detect_trampoline {hex(func_ea)}: failed to decode insn1 at {hex(next_ea)}")
            return None

        # Instruction 0 must be MOV DPTR, #imm
        mnem0 = idc.print_insn_mnem(func_ea).upper()
        dbg("trampolines", f"_detect_trampoline {hex(func_ea)}: insn0={mnem0} op0.type={insn0.ops[0].type} op0.reg={insn0.ops[0].reg} op1.type={insn0.ops[1].type}")
        if mnem0 != "MOV":
            dbg("trampolines", f"  insn0 mnem {mnem0!r} != MOV")
            return None
        if not (insn0.ops[0].type == ida_ua.o_reg
                and insn0.ops[0].reg == REG_DPTR
                and insn0.ops[1].type == ida_ua.o_imm):
            dbg("trampolines", f"  insn0 not MOV DPTR,#imm (o_reg={ida_ua.o_reg} REG_DPTR={REG_DPTR} o_imm={ida_ua.o_imm})")
            return None

        # Instruction 1 must be unconditional jump
        mnem1 = idc.print_insn_mnem(next_ea).upper()
        dbg("trampolines", f"  insn1 @{hex(next_ea)}: {mnem1} op0.type={insn1.ops[0].type}")
        if mnem1 not in ("LJMP", "SJMP", "AJMP"):
            dbg("trampolines", f"  insn1 mnem {mnem1!r} not a jump")
            return None
        if insn1.ops[0].type not in (ida_ua.o_near, ida_ua.o_far):
            dbg("trampolines", f"  insn1 op0.type {insn1.ops[0].type} not near/far")
            return None

        # Resolve helper EA (same-page address)
        page_base = func_ea & ~0xFFFF
        helper_ea = page_base | (insn1.ops[0].addr & 0xFFFF)
        dbg("trampolines", f"  helper_ea={hex(helper_ea)} (page_base={hex(page_base)}, op.addr={hex(insn1.ops[0].addr)})")

        # Verify helper is a page-switch helper
        page_num = _detect_page_in_func(helper_ea)
        dbg("trampolines", f"  page_num={page_num}")
        if page_num is None:
            return None

        # Resolve target: prefer symbolic name from IDA
        raw_operand = idc.print_operand(func_ea, 1)           # e.g. "#something_osd_0"
        dbg("trampolines", f"  raw_operand={raw_operand!r}")
        if raw_operand.startswith('#'):
            raw_operand = raw_operand[1:]

        target_ea: Optional[int] = None
        if raw_operand and not raw_operand[0].isdigit() and not raw_operand.startswith('0'):
            ea = ida_name.get_name_ea(BADADDR, raw_operand)
            dbg("trampolines", f"  symbol {raw_operand!r} → EA={hex(ea) if ea != BADADDR else 'BADADDR'}")
            if ea != BADADDR:
                target_ea = ea

        if target_ea is None:
            # Fallback: use page number to locate target
            offset = insn0.ops[1].value & 0xFFFF
            seg_base = _page_segment_base(page_num)
            target_ea = seg_base | offset
            dbg("trampolines", f"  fallback target_ea={hex(target_ea)} (page={page_num}, offset={hex(offset)}, seg_base={hex(seg_base)})")

        # Prefer a verified function entry, but accept any named/code address.
        # When MOV DPTR, #0xCF4B uses a raw hex offset and no IDA function has
        # been defined there yet, we still resolve to that EA rather than falling
        # back to inlining the trampoline body.
        target_fn = ida_funcs.get_func(target_ea)
        if target_fn is not None and target_fn.start_ea != target_ea:
            dbg("trampolines", f"  target_ea={hex(target_ea)} is inside function {hex(target_fn.start_ea)}, using containing function")
            target_ea = target_fn.start_ea

        dbg("trampolines", f"  → trampoline for {hex(target_ea)}")
        return target_ea

    except Exception:
        dbg("trampolines", f"_detect_trampoline {hex(func_ea)} exception: {traceback.format_exc()}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_trampoline_target(func_ea: int) -> Optional[int]:
    """
    Return the real target function EA if func_ea is a cross-page trampoline.
    Returns None if func_ea is not a trampoline or detection fails.
    Results are cached per func_ea.
    """
    if func_ea in _cache:
        return _cache[func_ea]
    result = _detect_trampoline(func_ea)
    _cache[func_ea] = result
    return result


def resolve_callee(callee_name: str) -> str:
    """
    If callee_name names a cross-page trampoline, return the real target's
    function name.  Otherwise return callee_name unchanged.
    """
    try:
        import idc, ida_name, ida_funcs
        ea = ida_name.get_name_ea(idc.BADADDR, callee_name)
        dbg("trampolines", f"resolve_callee({callee_name!r}): EA={hex(ea) if ea != idc.BADADDR else 'BADADDR'}")
        if ea == idc.BADADDR:
            return callee_name
        real_ea = get_trampoline_target(ea)
        if real_ea is None:
            return callee_name
        result = (ida_funcs.get_func_name(real_ea)
                  or ida_name.get_name(real_ea)
                  or hex(real_ea))
        dbg("trampolines", f"  → resolved to {result!r}")
        return result
    except Exception:
        dbg("trampolines", f"resolve_callee({callee_name!r}) exception: {traceback.format_exc()}")
        return callee_name


def clear_cache() -> None:
    """Clear the trampoline detection cache (call after IDA database changes)."""
    _cache.clear()
