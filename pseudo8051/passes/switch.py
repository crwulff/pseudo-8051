"""
passes/switch.py — SwitchStructurer: detect 8051 switch patterns.

The 8051 compiler emits switch-like dispatch as a chain of cumulative ADD + JZ/JNZ:

    mov  A, R7
    add  A, #-2     ; A = R7 - 2
    jz   label_2    ; if R7 == 2
    add  A, #-2     ; cumulative: R7 - 4
    jz   label_4    ; if R7 == 4
    add  A, #-4     ; cumulative: R7 - 8
    jnz  label_def  ; if R7 != 8 → default

Each (add, jz/jnz) pair is a separate basic block. This pass detects chains of
≥ 2 such blocks and replaces the head block's HIR with a SwitchNode.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import (
    HIRNode, Label, Assign, CompoundAssign, IfGoto, SwitchNode)
from pseudo8051.ir.expr import Reg, Const, BinOp, Expr, UnaryOp
from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock
from pseudo8051.passes        import OptimizationPass
from pseudo8051.constants     import dbg


def _label_for(block: BasicBlock) -> str:
    return block.label or f"label_{hex(block.start_ea).removeprefix('0x')}"


def _contains_a(expr: Expr) -> bool:
    """True if the expression tree contains a reference to register A."""
    if isinstance(expr, Reg):
        return expr.name == "A"
    if isinstance(expr, BinOp):
        return _contains_a(expr.lhs) or _contains_a(expr.rhs)
    if isinstance(expr, UnaryOp):
        return _contains_a(expr.operand)
    return False


def _extract_switch_step(block: BasicBlock):
    """
    Detect whether block matches the single switch-step pattern:

        [optional: Assign(A, subj)]
        CompoundAssign(A, +=, K)
        IfGoto(BinOp(A, ==|!=, 0), label)   ← must be last node

    Returns (delta, label, is_ne, subject) or None.
    - delta:   the byte constant added to A (0–255)
    - label:   the jump target label string
    - is_ne:   True for jnz (jump on A != 0)
    - subject: Expr from Assign(A, subj) if found, else None
    """
    hir = [n for n in block.hir if not isinstance(n, Label)]

    # Required: IfGoto(BinOp(A, op, 0), label) — must be the LAST non-Label node.
    # We scan from the end so that preamble code in the head block is ignored.
    if not hir:
        return None
    ig = hir[-1]
    if not (isinstance(ig, IfGoto)
            and isinstance(ig.cond, BinOp)
            and ig.cond.lhs == Reg("A")
            and ig.cond.rhs == Const(0)
            and ig.cond.op in ("==", "!=")):
        return None

    # Required: CompoundAssign(A, +=, K) — second-to-last node.
    if len(hir) < 2:
        return None
    ca = hir[-2]
    if not (isinstance(ca, CompoundAssign)
            and ca.lhs == Reg("A")
            and ca.op == "+="
            and isinstance(ca.rhs, Const)):
        return None
    delta = ca.rhs.value

    # Optional: Assign(A, subj) — third-to-last, where subj doesn't reference A.
    subject = None
    if len(hir) >= 3:
        maybe = hir[-3]
        if (isinstance(maybe, Assign)
                and maybe.lhs == Reg("A")
                and not _contains_a(maybe.rhs)):
            subject = maybe.rhs

    dbg("switch", f"  ✓ step @ {hex(block.start_ea)}: "
                  f"delta={hex(delta)} label={ig.label!r} "
                  f"is_ne={ig.cond.op=='!='} subj={subject!r}")
    return (delta, ig.label, ig.cond.op == "!=", subject)


def _fall_through_successor(block: BasicBlock,
                             jump_label: str) -> Optional[BasicBlock]:
    """Return the successor block whose derived label does NOT match jump_label."""
    for succ in block.successors:
        if _label_for(succ) != jump_label:
            return succ
    return None


def _try_switch(func: Function, head_block: BasicBlock) -> bool:
    """
    Attempt to build a SwitchNode from the chain starting at head_block.
    Returns True if a SwitchNode was created and blocks were absorbed.
    """
    steps: List[Tuple[int, str, bool, BasicBlock]] = []
    subj: Optional[Expr] = None
    cur = head_block

    while True:
        result = _extract_switch_step(cur)
        if result is None:
            break
        delta, label, is_ne, subject = result
        if subject is not None:
            subj = subject
        steps.append((delta, label, is_ne, cur))
        if is_ne:
            break   # jnz terminates the chain
        # Follow fall-through successor to the next step
        nxt = _fall_through_successor(cur, label)
        if nxt is None or getattr(nxt, "_absorbed", False):
            break
        cur = nxt

    if len(steps) < 2 or subj is None:
        return False

    # Build cases: compute cumulative offsets and derive the value of subj that
    # makes A == 0 at each step.
    cases_dict: Dict[str, List[int]] = {}   # label → [case values, ...]
    cases_order: List[str] = []             # insertion order
    default_label: Optional[str] = None
    cumulative = 0

    for delta, label, is_ne, blk in steps:
        cumulative = (cumulative + delta) & 0xFF
        case_val   = (-cumulative) & 0xFF

        if is_ne:
            # jnz: case_val is the fall-through value; label is the default
            default_label = label
            ft = _fall_through_successor(blk, label)
            if ft is not None:
                ft_label = _label_for(ft)
                if ft_label not in cases_dict:
                    cases_dict[ft_label] = []
                    cases_order.append(ft_label)
                cases_dict[ft_label].append(case_val)
        else:
            if label not in cases_dict:
                cases_dict[label] = []
                cases_order.append(label)
            cases_dict[label].append(case_val)

    cases = [(cases_dict[lbl], lbl) for lbl in cases_order]

    dbg("switch", f"block {hex(head_block.start_ea)}: "
                  f"SwitchNode subj={subj.render()!r} "
                  f"{len(steps)} steps → {len(cases)} case entries")

    first_hir = next((n for n in head_block.hir if not isinstance(n, Label)), None)
    sw_ea = first_hir.ea if first_hir is not None else head_block.start_ea
    sw = SwitchNode(
        ea            = sw_ea,
        subject       = subj,
        cases         = cases,
        default_label = default_label,
    )

    # Keep any Label nodes and all preamble code that precedes the switch tail.
    # The tail is: [Assign(A,subj)?] + CompoundAssign + IfGoto — 2 or 3 nodes.
    hir_no_labels = [n for n in head_block.hir if not isinstance(n, Label)]
    tail_len = 2   # CompoundAssign + IfGoto always consumed
    if (len(hir_no_labels) >= 3
            and isinstance(hir_no_labels[-3], Assign)
            and hir_no_labels[-3].lhs == Reg("A")
            and not _contains_a(hir_no_labels[-3].rhs)):
        tail_len = 3   # also consume the Assign(A, subj)
    preamble   = hir_no_labels[:-tail_len]
    label_nodes = [n for n in head_block.hir if isinstance(n, Label)]
    head_block.hir = label_nodes + preamble + [sw]

    # Absorb all intermediate blocks (all steps except the head block)
    for _, _, _, blk in steps[1:]:
        blk._absorbed = True

    return True


class SwitchStructurer(OptimizationPass):
    """
    Detect 8051 switch chains (cumulative ADD + JZ/JNZ) and replace with SwitchNode.
    Runs before IfElseStructurer so the chain blocks are absorbed before if/else
    structuring tries to process them.
    """

    def run(self, func: Function) -> None:
        changed = True
        while changed:
            changed = False
            for block in func.blocks:
                if getattr(block, "_absorbed", False):
                    continue
                if _try_switch(func, block):
                    changed = True
                    break   # restart — absorbed-set has changed
