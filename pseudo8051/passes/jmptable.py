"""
passes/jmptable.py — JmpTableStructurer: detect JMP @A+DPTR switch dispatch.

The 8051 compiler emits indirect-jump switch dispatch as:

    mov  DPTR, #code_7_1326    ; table base
    jmp  @A+DPTR               ; dispatch: PC = A + DPTR

where the table at code_7_1326 contains N equal-size unconditional jumps:

    code_7:1326  sjmp  case_0_handler
    code_7:1328  sjmp  case_1_handler   (stride=2 for SJMP, 3 for LJMP)

IDA adds the SJMP blocks as function tails and creates CFG edges from the
JMP @A+DPTR block to each SJMP entry.  This pass detects the pattern and
builds a SwitchNode, absorbing the SJMP intermediary blocks.

Detection uses two complementary strategies:

  CFG strategy (fast): use block.successors — the SJMP entries are in
    _block_map and IDA created explicit CFG edges.

  Memory strategy (fallback): read the table base from DPTR (via CPState),
    then decode SJMP/LJMP entries directly from code memory.  Needed when
    the dispatch blocks are owned by a different IDA function entry and were
    therefore filtered out of _block_map.
"""

import math
from typing import List, Optional, Tuple

import ida_ua
import ida_name

from pseudo8051.ir.hir         import (HIRNode, Label, Assign,
                                        SwitchNode, ComputedJump)
from pseudo8051.ir.expr        import Reg, Regs, Const, BinOp, Expr
from pseudo8051.ir.function    import Function
from pseudo8051.ir.basicblock  import BasicBlock
from pseudo8051.passes         import OptimizationPass
from pseudo8051.constants      import dbg


def _label_for(block: BasicBlock) -> str:
    return block.label or f"label_{hex(block.start_ea).removeprefix('0x')}"


# ── Locate the JMP @A+DPTR HIR node ──────────────────────────────────────────

def _find_jmp_hir_idx(block: BasicBlock) -> Optional[int]:
    """
    Return the index in block.hir of the HIR node produced by the JMP @A+DPTR
    instruction, or None if not present.

    Two strategies:
    1. Text match on Statement — works when JmpAtADptrHandler is in use.
    2. Instruction mnemonic scan + EA match — works even when DefaultHandler
       fired (e.g. module not fully reloaded) and produced '/* JMP @A+... */'.
    """
    # Strategy 1: ComputedJump node produced by JmpAtADptrHandler
    for i, node in enumerate(block.hir):
        if isinstance(node, ComputedJump):
            return i

    # Strategy 2: find the JMP instruction by mnemonic, then match its EA
    jmp_ea = None
    for insn in block.instructions:
        if insn.mnemonic == "JMP":
            jmp_ea = insn.ea
            break
    if jmp_ea is None:
        return None
    for i, node in enumerate(block.hir):
        if hasattr(node, "ea") and node.ea == jmp_ea:
            if isinstance(node, SwitchNode):
                return None  # already converted — don't process again
            return i
    return None


# ── Find preceding MOV DPTR assign ───────────────────────────────────────────

def _find_preceding_dptr_assign(hir: List[HIRNode], jmp_idx: int) -> Optional[int]:
    """
    Walk backwards from jmp_idx-1, up to 3 steps, looking for
    Assign(DPTR, ...).  Return the index, or None if not found.
    """
    limit = max(0, jmp_idx - 3)
    for i in range(jmp_idx - 1, limit - 1, -1):
        node = hir[i]
        if isinstance(node, Label):
            continue
        if (isinstance(node, Assign)
                and node.lhs == Reg("DPTR")):
            return i
        # Any intervening non-trivial node stops the search
        break
    return None


# ── Get jump-table EA via CPState ─────────────────────────────────────────────

def _get_table_ea(block: BasicBlock, jmp_idx: int) -> Optional[int]:
    """
    Determine the jump-table base address (full EA).

    Tries in order:
    1. Read rhs of the Assign(DPTR, Const(addr)) immediately before jmp_idx.
    2. Thread CPState through the block to the JMP instruction.

    Returns the full IDA EA, or None.
    """
    page_base = block.start_ea & ~0xFFFF

    # Strategy 1: HIR Assign
    dptr_idx = _find_preceding_dptr_assign(block.hir, jmp_idx)
    if dptr_idx is not None:
        node = block.hir[dptr_idx]
        if isinstance(node, Assign) and isinstance(node.rhs, Const):
            return page_base | (node.rhs.value & 0xFFFF)

    # Strategy 2: CPState threading
    if block.cp_entry is None:
        return None
    from pseudo8051.ir.cpstate import propagate_insn
    state = block.cp_entry.copy()
    jmp_ea = block.hir[jmp_idx].ea if hasattr(block.hir[jmp_idx], "ea") else None
    for insn in block.instructions:
        if jmp_ea is not None and insn.ea >= jmp_ea:
            break
        raw = insn.insn
        if raw is not None:
            propagate_insn(raw, state)
    dptr_val = state.get("DPTR")
    if dptr_val is not None:
        return page_base | (dptr_val & 0xFFFF)
    return None


# ── Read jump table from code memory ─────────────────────────────────────────

def _read_jump_table(table_ea: int, page_base: int, func: Function,
                     max_entries: int = 256) -> Tuple[List[Tuple[List[int], str]], int]:
    """
    Decode unconditional branch instructions at table_ea.

    Returns (cases, stride) where cases is [(value_list, target_label), ...].
    Returns ([], 0) if the table cannot be read or has < 2 valid entries.
    """
    entries: List[Tuple[int, str]] = []   # (entry_ea, target_label)
    ea = table_ea
    expected_stride: Optional[int] = None

    for _ in range(max_entries):
        insn = ida_ua.insn_t()
        size = ida_ua.decode_insn(insn, ea)
        if size == 0:
            break
        mnem = insn.get_canon_mnem().upper()
        if mnem not in ("SJMP", "LJMP", "AJMP"):
            break
        if expected_stride is None:
            expected_stride = size
        elif size != expected_stride:
            break   # inconsistent stride → end of table

        # Extract branch target
        target: Optional[int] = None
        for i in range(6):
            op = insn.ops[i]
            if op.type in (ida_ua.o_near, ida_ua.o_far):
                target = page_base | (op.addr & 0xFFFF)
                break
        if target is None:
            break

        # Resolve to a label: prefer block label, then IDA name, then generated
        tgt_block = func._block_map.get(target)
        if tgt_block is not None:
            lbl = _label_for(tgt_block)
        else:
            ida_lbl = ida_name.get_name(target)
            lbl = ida_lbl if ida_lbl else f"label_{hex(target).removeprefix('0x')}"

        entries.append((ea, lbl))
        ea += size

    if len(entries) < 2:
        return [], 0

    stride = expected_stride or 1

    cases: List[Tuple[List[int], str]] = []
    for i, (_entry_ea, lbl) in enumerate(entries):
        cases.append(([i], lbl))

    return cases, stride


# ── CFG-based case extraction ─────────────────────────────────────────────────

def _cases_from_cfg(block: BasicBlock) -> Tuple[Optional[List], Optional[int]]:
    """
    Use block.successors (SJMP dispatch entries) to build cases.

    Returns (cases, stride) or (None, None) on failure.
    """
    succs = [s for s in block.successors
             if not getattr(s, "_absorbed", False)]
    if len(succs) < 2:
        return None, None

    succs_sorted = sorted(succs, key=lambda b: b.start_ea)
    stride = succs_sorted[1].start_ea - succs_sorted[0].start_ea

    cases: List[Tuple[List[int], str]] = []
    for i, sjmp_blk in enumerate(succs_sorted):
        inner = [s for s in sjmp_blk.successors
                 if not getattr(s, "_absorbed", False)]
        if len(inner) != 1:
            dbg("switch", f"JmpTableStructurer(CFG): SJMP block "
                          f"{hex(sjmp_blk.start_ea)} has {len(inner)} "
                          f"inner successors — aborting CFG path")
            return None, None
        lbl = _label_for(inner[0])
        cases.append(([i], lbl))

    return cases, stride


# ── Main detection ────────────────────────────────────────────────────────────

def _try_jmptable(func: Function, block: BasicBlock) -> bool:
    """
    Attempt to build a SwitchNode from a JMP @A+DPTR block.
    Returns True if a SwitchNode was created.
    """
    jmp_idx = _find_jmp_hir_idx(block)
    if jmp_idx is None:
        return False

    # ── Strategy 1: CFG successors ────────────────────────────────────────
    cases, stride = _cases_from_cfg(block)

    if cases is None:
        dbg("switch", f"JmpTableStructurer: block {hex(block.start_ea)} "
                      f"CFG gave {len(block.successors)} successor(s) — "
                      f"trying memory fallback")

        # ── Strategy 2: read table from code memory ───────────────────────
        table_ea = _get_table_ea(block, jmp_idx)
        if table_ea is None:
            dbg("switch", f"JmpTableStructurer: block {hex(block.start_ea)} "
                          f"cannot determine DPTR value — skipping")
            return False

        page_base = block.start_ea & ~0xFFFF
        cases, stride = _read_jump_table(table_ea, page_base, func)
        if len(cases) < 2:
            dbg("switch", f"JmpTableStructurer: block {hex(block.start_ea)} "
                          f"memory read at {hex(table_ea)} found "
                          f"{len(cases)} entries — skipping")
            return False

        dbg("switch", f"JmpTableStructurer(mem): block {hex(block.start_ea)} "
                      f"table @ {hex(table_ea)} → {len(cases)} cases "
                      f"stride={stride}")

        # Absorb the SJMP dispatch-table blocks by their table EAs.
        # block.successors is empty when IDA did not create CFG edges from
        # the JMP block, so we must look up the entries directly.
        for i in range(len(cases)):
            entry_ea = table_ea + i * stride
            sjmp_blk = func._block_map.get(entry_ea)
            if sjmp_blk is not None:
                sjmp_blk._absorbed = True
    else:
        # CFG path: absorb the SJMP dispatch blocks
        succs_sorted = sorted(
            [s for s in block.successors if not getattr(s, "_absorbed", False)],
            key=lambda b: b.start_ea)
        for sjmp_blk in succs_sorted:
            sjmp_blk._absorbed = True

    # ── Build SwitchNode ──────────────────────────────────────────────────
    # Case values are always 0, 1, 2, … (the logical switch index).
    # The subject expression divides A by the stride to recover the index:
    #   stride=1 → A           (no division needed)
    #   stride=2 → A >> 1      (power-of-2: shift is cheaper than divide)
    #   stride=4 → A >> 2
    #   stride=3 → A / 3       (non-power-of-2: explicit divide)
    subject: Expr = Reg("A")
    if stride > 1:
        is_pow2 = (stride & (stride - 1)) == 0
        if is_pow2:
            shift = int(math.log2(stride))
            subject = BinOp(Reg("A"), ">>", Const(shift))
        else:
            subject = BinOp(Reg("A"), "/", Const(stride))

    dptr_idx = _find_preceding_dptr_assign(block.hir, jmp_idx)
    start = dptr_idx if dptr_idx is not None else jmp_idx

    sw_ea = block.hir[jmp_idx].ea
    sw = SwitchNode(sw_ea, subject, cases)
    sw.ann = block.hir[jmp_idx].ann  # carry forward annotation (reg_exprs, etc.)

    dbg("switch", f"JmpTableStructurer: block {hex(block.start_ea)} → "
                  f"SwitchNode subj={subject.render()!r} "
                  f"{len(cases)} cases stride={stride}")

    block.hir[start : jmp_idx + 1] = [sw]
    return True


def fixup_jmptable_edges(func: Function) -> None:
    """
    For every JMP @A+DPTR block whose IDA successors are empty, read the
    jump table from code memory and inject synthetic CFG edges so that
    passes running before JmpTableStructurer see a correct graph.

    Requires block.hir to be populated (runs after initial_hir()).
    """
    for block in func.blocks:
        if block.successors:          # IDA already has edges — nothing to do
            continue
        jmp_idx = _find_jmp_hir_idx(block)
        if jmp_idx is None:
            continue
        table_ea = _get_table_ea(block, jmp_idx)
        if table_ea is None:
            continue
        page_base = block.start_ea & ~0xFFFF
        cases, stride = _read_jump_table(table_ea, page_base, func)
        if not cases:
            continue
        dbg("switch", f"fixup_jmptable_edges: block {hex(block.start_ea)} "
                      f"table @ {hex(table_ea)} → {len(cases)} edges stride={stride}")
        for i in range(len(cases)):
            entry_ea = table_ea + i * stride
            sjmp_blk = func._block_map.get(entry_ea)
            if sjmp_blk is not None:
                block._add_successor(sjmp_blk)
                dbg("switch", f"  {hex(block.start_ea)} → {hex(sjmp_blk.start_ea)}")


class JmpTableStructurer(OptimizationPass):
    """
    Detect JMP @A+DPTR indirect-jump switch dispatch and replace with SwitchNode.
    Runs before SwitchStructurer so the SJMP blocks are absorbed first.
    """

    def run(self, func: Function) -> None:
        changed = True
        while changed:
            changed = False
            for block in func.blocks:
                if getattr(block, "_absorbed", False):
                    continue
                if _try_jmptable(func, block):
                    changed = True
                    break  # restart — absorbed-set has changed
        from pseudo8051.passes.debug_dump import dump_pass_hir
        all_nodes = [n for b in func.blocks
                     if not getattr(b, "_absorbed", False) for n in b.hir]
        dump_pass_hir("04.jmptable", all_nodes, func.name)
