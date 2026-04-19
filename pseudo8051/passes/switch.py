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

from typing import Dict, List, Optional, Tuple, Union

from pseudo8051.ir.hir import (
    HIRNode, Label, Assign, CompoundAssign, ExprStmt, IfGoto, SwitchNode,
    GotoStatement, BreakStmt, IfNode, ReturnStmt)
from pseudo8051.ir.expr import Reg, Regs, Const, BinOp, Expr, UnaryOp
from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock
from pseudo8051.passes        import OptimizationPass, run_blocks_until_stable, dump_hir
from pseudo8051.constants     import dbg


def _body_text(nodes: List[HIRNode]) -> List[str]:
    """Return flat list of rendered text lines for a case body (for deduplication)."""
    return [t for n in nodes for _, t in n.render()]


def _label_for(block: BasicBlock) -> str:
    return block.label or f"label_{hex(block.start_ea).removeprefix('0x')}"


def _contains_a(expr: Expr) -> bool:
    """True if the expression tree contains a reference to register A."""
    if isinstance(expr, Regs) and expr.is_single:
        return expr == Reg("A")
    if isinstance(expr, BinOp):
        return _contains_a(expr.lhs) or _contains_a(expr.rhs)
    if isinstance(expr, UnaryOp):
        return _contains_a(expr.operand)
    return False


def _extract_a_delta(node) -> Optional[int]:
    """
    If node modifies A by a compile-time constant, return that constant mod 256.
    Recognises:
      CompoundAssign(A, +=, K)  → K
      CompoundAssign(A, -=, K)  → (−K) & 0xFF
      ExprStmt(A--)             → 0xFF  (dec A)
      ExprStmt(A++)             → 0x01  (inc A)
    Returns None if the node doesn't match any of these.
    """
    if isinstance(node, CompoundAssign) and node.lhs == Reg("A") and isinstance(node.rhs, Const):
        if node.op == "+=":
            return node.rhs.value & 0xFF
        if node.op == "-=":
            return (-node.rhs.value) & 0xFF
    if isinstance(node, ExprStmt) and isinstance(node.expr, UnaryOp):
        uo = node.expr
        if isinstance(uo.operand, Regs) and uo.operand == Reg("A"):
            if uo.op == "--":
                return 0xFF
            if uo.op == "++":
                return 0x01
    return None


def _extract_switch_step(block: BasicBlock):
    """
    Detect whether block matches the single switch-step pattern:

        [optional: Assign(A, subj)]
        <A-delta node>
        IfGoto(BinOp(A, ==|!=, 0), label)   ← must be last node

    The A-delta node may be any of:
        CompoundAssign(A, +=, K)   — compiler ADD
        CompoundAssign(A, -=, K)   — compiler SUBB
        ExprStmt(A--)              — compiler DEC A
        ExprStmt(A++)              — compiler INC A

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

    # Required: A-delta node — second-to-last.
    if len(hir) < 2:
        return None
    delta = _extract_a_delta(hir[-2])
    if delta is None:
        return None

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
                # If this block has no IDA-assigned label, _label_for() returns
                # a synthetic "label_XXXX" name, but initial_hir() never inserted
                # a Label HIR node (it only does so when block.label is set).
                # Assign the label now so that _absorb_switch_in_body_list can
                # find the arm boundary when the loop body is assembled flat.
                if ft.label is None:
                    ft.label = ft_label
                    ft.hir.insert(0, Label(ft.start_ea, ft_label))
                if ft_label not in cases_dict:
                    cases_dict[ft_label] = []
                    cases_order.append(ft_label)
                cases_dict[ft_label].append(case_val)
        else:
            if label not in cases_dict:
                cases_dict[label] = []
                cases_order.append(label)
            cases_dict[label].append(case_val)

    cases = [(sorted(cases_dict[lbl]), lbl) for lbl in cases_order]
    cases.sort(key=lambda pair: min(pair[0]))

    # If all steps used JEZ (is_ne=False) the chain has no explicit default branch.
    # The fall-through from the last step is the default arm — capture it now so
    # that _absorb_switch_in_body_list can find and partition the arm correctly.
    if default_label is None and steps:
        _, last_lbl, last_is_ne, last_blk = steps[-1]
        if not last_is_ne:
            ft_def = _fall_through_successor(last_blk, last_lbl)
            if ft_def is not None:
                default_label = _label_for(ft_def)
                if ft_def.label is None:
                    ft_def.label = default_label
                    ft_def.hir.insert(0, Label(ft_def.start_ea, default_label))

    dbg("switch", f"block {hex(head_block.start_ea)}: "
                  f"SwitchNode subj={subj.render()!r} "
                  f"{len(steps)} steps → {len(cases)} case entries")

    # ── Per-case EA tracking ──────────────────────────────────────────────────
    # Map each case label to the union of src_eas from all steps that jump there.
    _step_eas_by_label: Dict[str, frozenset] = {}
    _default_step_eas: frozenset = frozenset()
    _all_step_eas: frozenset = frozenset()

    for delta, label, is_ne, blk in steps:
        _hir_nl = [n for n in blk.hir if not isinstance(n, Label)]
        _tl = 2
        if (len(_hir_nl) >= 3
                and isinstance(_hir_nl[-3], Assign)
                and _hir_nl[-3].lhs == Reg("A")
                and not _contains_a(_hir_nl[-3].rhs)):
            _tl = 3
        _step_ea_set = frozenset().union(*(n.src_eas for n in _hir_nl[-_tl:]))
        _all_step_eas |= _step_ea_set
        if is_ne:
            _default_step_eas |= _step_ea_set
            _ft = _fall_through_successor(blk, label)
            if _ft is not None:
                _ft_lbl = _label_for(_ft)
                _step_eas_by_label[_ft_lbl] = (
                    _step_eas_by_label.get(_ft_lbl, frozenset()) | _step_ea_set)
        else:
            _step_eas_by_label[label] = (
                _step_eas_by_label.get(label, frozenset()) | _step_ea_set)

    first_hir = next((n for n in head_block.hir if not isinstance(n, Label)), None)
    sw_ea = first_hir.ea if first_hir is not None else head_block.start_ea
    sw = SwitchNode(
        ea              = sw_ea,
        subject         = subj,
        cases           = cases,
        default_label   = default_label,
        case_src_eas    = [_step_eas_by_label.get(lbl, frozenset()) for _, lbl in cases],
        default_src_eas = _default_step_eas if _default_step_eas else None,
    )
    sw.src_eas = _all_step_eas if _all_step_eas else sw.src_eas

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


def _extract_linear_equality_step(block: BasicBlock):
    """
    Like _extract_switch_step but also handles CPL (A=~A) and XRL (A^=K).

    Returns (transform, label, is_ne, subject, did_reload) or None.
    - transform: ('add', K) for additive delta, ('xor', K) for XOR/CPL (XOR 0xFF)
    - label: IfGoto branch target
    - is_ne: True for jnz (default arm)
    - subject: Expr from Assign(A, subj) reload if found, else None
    - did_reload: True if the block reloaded A from a subject before the transform
    """
    hir = [n for n in block.hir if not isinstance(n, Label)]
    if not hir:
        return None

    # Must end with IfGoto(BinOp(A, op, 0))
    ig = hir[-1]
    if not (isinstance(ig, IfGoto)
            and isinstance(ig.cond, BinOp)
            and ig.cond.lhs == Reg("A")
            and ig.cond.rhs == Const(0)
            and ig.cond.op in ("==", "!=")):
        return None

    if len(hir) < 2:
        return None

    delta_node = hir[-2]
    transform = None

    # ADD/SUB/XOR via CompoundAssign(A, op=, K)
    if isinstance(delta_node, CompoundAssign) and delta_node.lhs == Reg("A"):
        if delta_node.op == "+=" and isinstance(delta_node.rhs, Const):
            transform = ('add', delta_node.rhs.value & 0xFF)
        elif delta_node.op == "-=" and isinstance(delta_node.rhs, Const):
            transform = ('add', (-delta_node.rhs.value) & 0xFF)
        elif delta_node.op == "^=" and isinstance(delta_node.rhs, Const):
            transform = ('xor', delta_node.rhs.value & 0xFF)
    # DEC/INC via ExprStmt(UnaryOp(op, A))
    elif isinstance(delta_node, ExprStmt) and isinstance(delta_node.expr, UnaryOp):
        uo = delta_node.expr
        if isinstance(uo.operand, Regs) and uo.operand == Reg("A"):
            if uo.op == "--":
                transform = ('add', 0xFF)   # dec A ≡ A += -1
            elif uo.op == "++":
                transform = ('add', 0x01)
    # CPL A: Assign(A, ~A)  →  XOR 0xFF
    elif (isinstance(delta_node, Assign)
          and delta_node.lhs == Reg("A")
          and isinstance(delta_node.rhs, UnaryOp)
          and delta_node.rhs.op == "~"
          and isinstance(delta_node.rhs.operand, Regs)
          and delta_node.rhs.operand == Reg("A")):
        transform = ('xor', 0xFF)

    if transform is None:
        return None

    # Optional reload: Assign(A, subj) as third-to-last node.
    # Must not be a CPL (A = ~A) — that's the transform node itself.
    subject = None
    did_reload = False
    if len(hir) >= 3:
        maybe = hir[-3]
        if (isinstance(maybe, Assign)
                and maybe.lhs == Reg("A")
                and not _contains_a(maybe.rhs)
                and not (isinstance(maybe.rhs, UnaryOp) and maybe.rhs.op == "~")):
            subject = maybe.rhs
            did_reload = True

    return (transform, ig.label, ig.cond.op == "!=", subject, did_reload)


def _try_linear_equality_switch(func: Function, head_block: BasicBlock) -> bool:
    """
    Detect a linear equality chain switch where each step uses CPL, XRL, ADD, DEC, or INC
    to test whether the switch subject equals a specific value.

    Unlike _try_switch (cumulative ADD + JZ/JNZ), this also handles:
      CPL A       (A = ~A)    → case value 0xFF
      XRL A, #K  (A ^= K)    → case value K  (with reload)

    Steps compose as: XOR chains stay XOR; ADD chains stay additive; reload resets.
    Mixed ADD-after-XOR (without reload) breaks the chain.

    Returns True if a SwitchNode was created and intermediate blocks absorbed.
    """
    steps: List[Tuple] = []   # (case_val, label, is_ne, block)
    subjects: List[Expr] = []
    cur = head_block

    # current_T tracks the net transform T applied to the subject since last reload.
    # T(x) = x + delta  (type 'add')  →  A==0 when x == (-delta)&0xFF
    # T(x) = x ^ mask   (type 'xor')  →  A==0 when x == mask
    # Identity is ('add', 0).
    current_T = ('add', 0)

    while True:
        result = _extract_linear_equality_step(cur)
        if result is None:
            break
        transform, label, is_ne, subject, did_reload = result

        if did_reload:
            current_T = ('add', 0)   # reset to identity on subject reload

        t_type, t_val = transform

        if current_T[0] == 'add' and t_type == 'add':
            new_T = ('add', (current_T[1] + t_val) & 0xFF)
        elif current_T[0] == 'add' and current_T[1] == 0 and t_type == 'xor':
            # Identity followed by XOR → pure XOR
            new_T = ('xor', t_val)
        elif current_T[0] == 'xor' and t_type == 'xor':
            # Chained XOR: mask accumulates
            new_T = ('xor', current_T[1] ^ t_val)
        else:
            # Mixed ADD+XOR without reload: case value not expressible as subj==K
            break

        current_T = new_T

        if current_T[0] == 'add':
            case_val = (-current_T[1]) & 0xFF
        else:
            case_val = current_T[1]

        if subject is not None:
            subjects.append(subject)

        steps.append((case_val, label, is_ne, cur, did_reload))
        dbg("switch", f"  leq step @ {hex(cur.start_ea)}: "
                      f"case={hex(case_val)} label={label!r} is_ne={is_ne}")

        if is_ne:
            break

        nxt = _fall_through_successor(cur, label)
        if nxt is None or getattr(nxt, "_absorbed", False):
            break
        cur = nxt

    if len(steps) < 2 or not subjects:
        return False

    # Only proceed if at least one step uses XOR/CPL (pure ADD chains are handled
    # by _try_switch which runs after this function).
    has_xor = any(
        _extract_linear_equality_step(blk) is not None
        and _extract_linear_equality_step(blk)[0][0] == 'xor'
        for _, _, _, blk, _ in steps
    )
    if not has_xor:
        return False

    subj = subjects[0]

    # Build cases dict (same logic as _try_switch)
    cases_dict: Dict[str, List[int]] = {}
    cases_order: List[str] = []
    default_label: Optional[str] = None

    for case_val, label, is_ne, blk, _dr in steps:
        if is_ne:
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

    # If chain ends with jz (no jnz), the fall-through of the last block is the default.
    if not any(is_ne for _, _, is_ne, _, _ in steps):
        last_blk = steps[-1][3]
        last_jz_label = steps[-1][1]
        ft_default = _fall_through_successor(last_blk, last_jz_label)
        if ft_default is not None:
            default_label = _label_for(ft_default)

    if not cases_dict:
        return False

    cases = [(sorted(cases_dict[lbl]), lbl) for lbl in cases_order]
    cases.sort(key=lambda pair: min(pair[0]))

    dbg("switch", f"block {hex(head_block.start_ea)}: "
                  f"LinearEqSwitch subj={subj.render()!r} "
                  f"{len(steps)} steps → {len(cases)} case entries")

    # ── Per-case EA tracking ──────────────────────────────────────────────────
    _step_eas_by_label: Dict[str, frozenset] = {}
    _default_step_eas: frozenset = frozenset()
    _all_step_eas: frozenset = frozenset()

    for case_val, label, is_ne, blk, dr in steps:
        _hir_nl = [n for n in blk.hir if not isinstance(n, Label)]
        _tl = 3 if dr else 2
        _step_ea_set = frozenset().union(*(n.src_eas for n in _hir_nl[-_tl:]))
        _all_step_eas |= _step_ea_set
        if is_ne:
            _default_step_eas |= _step_ea_set
            _ft = _fall_through_successor(blk, label)
            if _ft is not None:
                _ft_lbl = _label_for(_ft)
                _step_eas_by_label[_ft_lbl] = (
                    _step_eas_by_label.get(_ft_lbl, frozenset()) | _step_ea_set)
        else:
            _step_eas_by_label[label] = (
                _step_eas_by_label.get(label, frozenset()) | _step_ea_set)

    first_hir = next((n for n in head_block.hir if not isinstance(n, Label)), None)
    sw_ea = first_hir.ea if first_hir is not None else head_block.start_ea
    sw = SwitchNode(
        ea              = sw_ea,
        subject         = subj,
        cases           = cases,
        default_label   = default_label,
        case_src_eas    = [_step_eas_by_label.get(lbl, frozenset()) for _, lbl in cases],
        default_src_eas = _default_step_eas if _default_step_eas else None,
    )
    sw.src_eas = _all_step_eas if _all_step_eas else sw.src_eas

    # Keep Label nodes and any preamble before the switch tail in head block.
    hir_no_labels = [n for n in head_block.hir if not isinstance(n, Label)]
    first_step_result = _extract_linear_equality_step(head_block)
    tail_len = 2 + (1 if first_step_result[4] else 0)   # +1 if reload node consumed
    preamble = hir_no_labels[:-tail_len]
    label_nodes = [n for n in head_block.hir if isinstance(n, Label)]
    head_block.hir = label_nodes + preamble + [sw]

    # Absorb all intermediate step blocks (not the head)
    for _, _, _, blk, _ in steps[1:]:
        blk._absorbed = True

    return True


class SwitchStructurer(OptimizationPass):
    """
    Detect 8051 switch chains and replace with SwitchNode.

    Two patterns are detected (in order):
    1. Linear equality chains using CPL/XRL/ADD/DEC — each step tests subject == K.
    2. Cumulative ADD/DEC chains (the original pattern).

    Runs before IfElseStructurer so chain blocks are absorbed before if/else structuring.
    """

    def run(self, func: Function) -> None:
        # Pass 1: linear equality chains (CPL/XRL — must run first to grab full chains)
        run_blocks_until_stable(func, _try_linear_equality_switch)
        # Pass 2: cumulative ADD chains
        run_blocks_until_stable(func, _try_switch)
        dump_hir(func, "05.switch")


# ── SwitchBodyAbsorber helpers ────────────────────────────────────────────────

def _arm_blocks_sw(start: BasicBlock, merge_ea: int) -> List[BasicBlock]:
    """
    Collect non-absorbed blocks belonging to one switch-case arm.

    Unlike _arm_blocks() from ifelse (which filters start_ea < merge_ea),
    this stops at the EXACT merge block so it works when the arm entry is
    at a higher EA than the merge (jump-table switches with reversed layout).

    Traverses absorbed blocks for CFG connectivity but does not include them
    in the result.
    """
    visited: set = set()
    result:  list = []
    queue = [start]
    while queue:
        blk = queue.pop(0)
        if blk.start_ea in visited:
            continue
        visited.add(blk.start_ea)
        if blk.start_ea == merge_ea:
            continue   # stop here — don't include or BFS past the merge block
        if not getattr(blk, "_absorbed", False):
            result.append(blk)
        for succ in blk.successors:
            if succ.start_ea not in visited:
                queue.append(succ)
    return result

def _merge_ea_from_labels(labels: List[str], label_to_block: dict,
                          branch_ea: int) -> Optional[int]:
    """
    Forward-BFS intersection over the given arm labels.
    Returns the smallest common reachable EA, or None.
    """
    from pseudo8051.passes.ifelse import _reachable_eas

    if not labels:
        return None
    sets = []
    for label in labels:
        blk = label_to_block.get(label)
        if blk is None:
            return None
        sets.append(_reachable_eas(blk, branch_ea))
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return min(common) if common else None


def _find_switch_merge_ea(func: Function, switch_node: SwitchNode,
                          branch_ea: int) -> Optional[int]:
    """
    Find the merge point (post-dominator) for a SwitchNode.

    Uses the minimum arm-block EA (minus 1) as the BFS floor so that
    jump-table switches (dispatch block AFTER case bodies in code order)
    can reach the merge block and trampolines.

    Strategy:
    1. Full intersection — merge reachable from ALL arms: return min(common).
    2. Mode fallback — some arms are dead-ends (return / jump out): return the
       EA reachable from the *maximum* number of arms.  Using the maximum count
       (rather than "any 2+ arms") avoids mistaking shared internal code blocks
       for the merge point.
    3. If no EA is shared by 2+ arms, all arms truly terminate → return None.
    """
    from pseudo8051.passes.ifelse import _reachable_eas
    from collections import Counter

    label_to_block = {_label_for(b): b for b in func.blocks}
    all_labels: List[str] = [body for _, body in switch_node.cases
                              if isinstance(body, str)]
    if switch_node.default_label:
        all_labels.append(switch_node.default_label)
    if not all_labels:
        return None

    arm_eas = [label_to_block[l].start_ea for l in all_labels if l in label_to_block]
    bfs_floor = min(arm_eas) - 1 if arm_eas else branch_ea

    per_arm: List[set] = []
    for label in all_labels:
        blk = label_to_block.get(label)
        if blk is None:
            # Arm target is outside this function (filtered by IDA as belonging
            # to a different function entry, e.g. an SJMP trampoline tail).
            # Skip it from the intersection — the merge can still be found from
            # the remaining arms.
            continue
        per_arm.append(_reachable_eas(blk, bfs_floor))

    if not per_arm:
        return None

    # Strategy 1: full intersection
    common = per_arm[0].copy()
    for s in per_arm[1:]:
        common &= s
    if common:
        return min(common)

    # Strategy 2: mode — EA reachable from the most arms
    ea_count: Counter = Counter(ea for s in per_arm for ea in s)
    max_cnt = max(ea_count.values(), default=0)
    if max_cnt < 2:
        return None  # all arms truly terminate independently
    candidates = {ea for ea, cnt in ea_count.items() if cnt == max_cnt}
    return min(candidates)


def _replace_goto_with_break(nodes: List[HIRNode],
                              merge_label: str) -> List[HIRNode]:
    """
    Recursively replace 'goto merge_label' with 'break;' throughout nodes.
    Does NOT recurse into WhileNode/ForNode bodies (break there targets the loop).
    """
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, GotoStatement) and node.label == merge_label:
            result.append(BreakStmt(node.ea))
        elif isinstance(node, IfGoto) and node.label == merge_label:
            result.append(IfNode(node.ea, node.cond, [BreakStmt(node.ea)]))
        elif isinstance(node, IfNode):
            node.then_nodes = _replace_goto_with_break(node.then_nodes,
                                                        merge_label)
            node.else_nodes = _replace_goto_with_break(node.else_nodes,
                                                        merge_label)
            result.append(node)
        else:
            result.append(node)
    return result


def _needs_break(body_nodes: List[HIRNode]) -> bool:
    """
    True if a trailing 'break;' should be appended (body doesn't already end
    with an unconditional exit).
    """
    if not body_nodes:
        return True
    last = body_nodes[-1]
    if isinstance(last, ReturnStmt):
        return False
    if isinstance(last, GotoStatement):
        return False
    if isinstance(last, BreakStmt):
        return False
    return True


# ── HIR-tree embedded switch absorption ──────────────────────────────────────
#
# When IfElseStructurer absorbs a switch block into an if-arm, it inlines the
# switch node and all case arm blocks as siblings in the arm's HIR body list.
# _build_arm_hir now preserves Label nodes that are switch case targets, so the
# body looks like:
#   [SwitchNode(goto labels), Label("A"), arm_A_nodes..., Label("B"), arm_B_nodes...]
#
# The functions below find such patterns in any HIR body list and absorb the
# label-delimited arm nodes into the SwitchNode's cases, mirroring _absorb().
# ─────────────────────────────────────────────────────────────────────────────

def _partition_by_switch_labels(sibling_nodes: List[HIRNode],
                                 case_labels: set):
    """
    Partition *sibling_nodes* (nodes that follow a SwitchNode in a body list)
    into:
      - outer_nodes: nodes before the first case label (stay in the outer body)
      - arm_groups: dict mapping case label → list of HIR nodes in that arm
    """
    outer_nodes: List[HIRNode] = []
    arm_groups: dict = {}
    current_label = None
    for sn in sibling_nodes:
        if isinstance(sn, Label) and sn.name in case_labels:
            current_label = sn.name
            arm_groups[current_label] = []
        elif current_label is None:
            outer_nodes.append(sn)
        else:
            arm_groups[current_label].append(sn)
    return outer_nodes, arm_groups


def _absorb_switch_in_body_list(nodes: List[HIRNode]) -> tuple:
    """
    Scan *nodes* for a SwitchNode whose case arms appear as Label-delimited
    sibling nodes (the signature left by IfElseStructurer when it absorbed the
    switch block together with its case arm blocks).  When found, absorb the
    arms into the SwitchNode and remove them from the body list.
    Returns (new_nodes, changed).
    """
    result: List[HIRNode] = []
    i = 0
    changed = False
    while i < len(nodes):
        node = nodes[i]
        if isinstance(node, SwitchNode):
            # Collect unresolved (string-label) case labels
            case_labels = {body for _, body in node.cases if isinstance(body, str)}
            if node.default_label and isinstance(node.default_label, str):
                case_labels.add(node.default_label)

            if case_labels:
                sibling_nodes = nodes[i + 1:]
                sibling_label_names = {
                    sn.name for sn in sibling_nodes if isinstance(sn, Label)
                }
                if case_labels & sibling_label_names:
                    outer_nodes, arm_groups = _partition_by_switch_labels(
                        sibling_nodes, case_labels)

                    # ── Pre-structuring ───────────────────────────────────────
                    # Structure flat if/else patterns in each arm BEFORE merge-
                    # point detection and cleanup.  This is required because:
                    #   1. Merge-point detection uses arm gotos to identify the
                    #      switch exit label, which must be present.
                    #   2. The cleanup step removes GotoStatement(merge_label)
                    #      from ALL arms, including ones embedded in diamond
                    #      patterns where that goto is semantically necessary.
                    # Running _structure_flat_ifelse now consumes those gotos
                    # into IfNodes before cleanup can break the diamonds.
                    from pseudo8051.passes.ifelse import _structure_flat_ifelse
                    for _lbl in arm_groups:
                        arm_groups[_lbl] = _structure_flat_ifelse(arm_groups[_lbl])

                    # ── Merge-point detection ─────────────────────────────────
                    # The merge label is the first non-case Label in the last
                    # arm's body that at least one other arm jumps to.  It marks
                    # the switch exit: everything from it onward belongs AFTER
                    # the switch, not inside any arm.
                    from pseudo8051.passes.ifelse import _collect_goto_targets
                    last_arm_lbl = list(arm_groups.keys())[-1]
                    other_gotos: set = set()
                    for _lbl, _arm_hir in arm_groups.items():
                        if _lbl != last_arm_lbl:
                            other_gotos |= _collect_goto_targets(_arm_hir)

                    merge_label: Optional[str] = None
                    post_switch_nodes: List[HIRNode] = []
                    last_arm_hir = arm_groups[last_arm_lbl]
                    for _idx, _n in enumerate(last_arm_hir):
                        if (isinstance(_n, Label)
                                and _n.name not in case_labels
                                and _n.name in other_gotos):
                            merge_label = _n.name
                            post_switch_nodes = last_arm_hir[_idx:]
                            arm_groups[last_arm_lbl] = last_arm_hir[:_idx]
                            break

                    # Fallback: other arms may fall through (no explicit goto) so
                    # other_gotos is empty or lacks the exit label.  Search ALL
                    # arms for the last non-case Label that is a goto target
                    # within that same arm — pre-structuring leaves GotoStatement
                    # nodes nested inside IfNodes pointing to this exit label.
                    if merge_label is None:
                        for _search_lbl in arm_groups:
                            _arm_hir_s = arm_groups[_search_lbl]
                            _arm_targets_s = _collect_goto_targets(_arm_hir_s)
                            for _idx in range(len(_arm_hir_s) - 1, -1, -1):
                                _n = _arm_hir_s[_idx]
                                if (isinstance(_n, Label)
                                        and _n.name not in case_labels
                                        and _n.name in _arm_targets_s):
                                    merge_label = _n.name
                                    post_switch_nodes = _arm_hir_s[_idx:]
                                    arm_groups[_search_lbl] = (
                                        _arm_hir_s[:_idx])
                                    break
                            if merge_label is not None:
                                break

                    # Replace goto(merge_label) with break throughout all arms,
                    # including ones nested inside IfNodes produced by pre-structuring.
                    # (A flat list-comprehension would miss gotos nested one level deep.)
                    if merge_label is not None:
                        for _lbl in arm_groups:
                            arm_groups[_lbl] = _replace_goto_with_break(
                                arm_groups[_lbl], merge_label)

                    # ── Embedded-tail resolution ──────────────────────────────
                    # Handles non-case Labels within arm bodies that represent
                    # shared code between arms (e.g. an SJMP-trampoline target
                    # whose code is laid out inline in the same arm).
                    # Strategy:
                    #   1. Collect every non-case Label found in any arm body.
                    #   2. Strip dead gotos: GotoStatement(L) immediately
                    #      followed by Label(L) — these are trampoline artifacts.
                    #   3. For any arm whose last node is 'goto shared_label',
                    #      strip the goto and append the shared tail in its place.
                    #   4. Remove the now-inlined embedded Label nodes.
                    embedded_tails: Dict[str, List[HIRNode]] = {}
                    for arm_nodes in arm_groups.values():
                        for idx, n in enumerate(arm_nodes):
                            if isinstance(n, Label) and n.name not in case_labels:
                                embedded_tails[n.name] = arm_nodes[idx + 1:]
                    if embedded_tails:
                        # Step 2: strip dead trampoline gotos
                        for lbl in arm_groups:
                            arm_hir = arm_groups[lbl]
                            cleaned: List[HIRNode] = []
                            for j, n in enumerate(arm_hir):
                                if (isinstance(n, GotoStatement)
                                        and n.label in embedded_tails
                                        and j + 1 < len(arm_hir)
                                        and isinstance(arm_hir[j + 1], Label)
                                        and arm_hir[j + 1].name == n.label):
                                    continue  # dead goto — target follows immediately
                                cleaned.append(n)
                            arm_groups[lbl] = cleaned
                        # Step 3: iteratively resolve end-gotos to embedded labels.
                        for lbl in arm_groups:
                            arm_hir = arm_groups[lbl]
                            while (arm_hir
                                   and isinstance(arm_hir[-1], GotoStatement)
                                   and arm_hir[-1].label in embedded_tails):
                                target_label = arm_hir[-1].label
                                arm_hir.pop()
                                arm_hir.extend(embedded_tails[target_label])
                        # Step 4: strip the now-inlined embedded Label nodes.
                        for lbl in arm_groups:
                            arm_groups[lbl] = [
                                n for n in arm_groups[lbl]
                                if not (isinstance(n, Label)
                                        and n.name in embedded_tails)
                            ]

                    # Structure flat if/else within each arm body before adding
                    # breaks.  The arm HIR is assembled from raw basic-block HIR
                    # that was never through if/else structuring (IfElseStructurer
                    # ran before SwitchBodyAbsorber).
                    from pseudo8051.passes.ifelse import _structure_flat_ifelse

                    # Build case bodies
                    new_cases: List = []
                    for values, body in node.cases:
                        if isinstance(body, str) and body in arm_groups:
                            arm_hir = _structure_flat_ifelse(list(arm_groups[body]))
                            if _needs_break(arm_hir):
                                arm_hir.append(BreakStmt(node.ea))
                            new_cases.append((values, arm_hir))
                        else:
                            new_cases.append((values, body))
                    node.cases = new_cases

                    # Dedup cases with identical bodies
                    deduped: List = []
                    for values, body in new_cases:
                        if isinstance(body, list):
                            for ev, eb in deduped:
                                if isinstance(eb, list) and (
                                        eb is body
                                        or _body_text(eb) == _body_text(body)):
                                    ev.extend(values)
                                    break
                            else:
                                deduped.append((list(values), body))
                        else:
                            deduped.append((list(values), body))
                    node.cases = deduped

                    # Handle default body
                    if (node.default_label
                            and isinstance(node.default_label, str)
                            and node.default_label in arm_groups):
                        dbody = _structure_flat_ifelse(list(arm_groups[node.default_label]))
                        if _needs_break(dbody):
                            dbody.append(BreakStmt(node.ea))
                        node.default_body = dbody
                        node.default_label = None

                    dbg("switch",
                        f"  _absorb_switch_in_body_list: switch @ {hex(node.ea)} "
                        f"absorbed {len(arm_groups)} arm(s) from HIR siblings")
                    result.append(node)
                    result.extend(outer_nodes)
                    result.extend(post_switch_nodes)  # merge point and beyond
                    i = len(nodes)   # skip all consumed siblings
                    changed = True
                    continue

        result.append(node)
        i += 1

    return result, changed


def _absorb_switches_in_node(node: HIRNode):
    """Recursively apply HIR-tree switch absorption to a node's bodies.
    Returns (modified_node, changed)."""
    changed = False

    def _fn(ns):
        nonlocal changed
        new_ns, ch = _absorb_switches_in_list(ns)
        if ch:
            changed = True
        return new_ns

    new_node = node.map_bodies(_fn)
    return new_node, changed


def _absorb_switches_in_list(nodes: List[HIRNode]) -> tuple:
    """
    Recursively absorb embedded SwitchNodes with Label-delimited sibling arms
    in a HIR body list.  Recurses depth-first into structured node bodies first,
    then handles the current list level.
    Returns (new_nodes, changed).
    """
    new_nodes: List[HIRNode] = []
    any_changed = False
    for node in nodes:
        node, ch = _absorb_switches_in_node(node)
        any_changed = any_changed or ch
        new_nodes.append(node)
    result, ch2 = _absorb_switch_in_body_list(new_nodes)
    return result, (any_changed or ch2)


class SwitchBodyAbsorber(OptimizationPass):
    """
    Absorb case body blocks directly into their SwitchNode.

    Must run after IfElseStructurer so case bodies are already fully structured
    (nested if/else built) before absorption.
    """

    def run(self, func: Function) -> None:
        from pseudo8051.passes.ifelse import (
            _build_arm_hir,
            _collect_goto_targets, _drop_dead_labels,
        )

        label_to_block = {_label_for(b): b for b in func.blocks}
        changed = False

        # ── CFG-based pass: SwitchNodes in non-absorbed blocks ────────────────
        for block in func.blocks:
            if getattr(block, "_absorbed", False):
                continue
            for node in block.hir:
                if not isinstance(node, SwitchNode):
                    continue
                if not any(isinstance(body, str) for _, body in node.cases):
                    continue  # already absorbed
                self._absorb(func, block, node, label_to_block,
                             _build_arm_hir, _collect_goto_targets)
                changed = True

        # ── HIR-tree pass: SwitchNodes embedded inside IfNode/loop bodies ─────
        # Occurs when IfElseStructurer absorbed the switch block together with
        # its case arm blocks.  The case arm Label nodes were preserved by
        # _build_arm_hir (keep_labels), so they appear as labeled siblings.
        hir_changed = False
        for block in func.blocks:
            if getattr(block, "_absorbed", False):
                continue
            new_hir, ch = _absorb_switches_in_list(block.hir)
            if ch:
                block.hir = new_hir
                hir_changed = True

        if changed or hir_changed:
            self._dead_label_cleanup(func, _collect_goto_targets, _drop_dead_labels)
        from pseudo8051.passes.debug_dump import dump_pass_hir
        all_nodes = [n for b in func.blocks
                     if not getattr(b, "_absorbed", False) for n in b.hir]
        dump_pass_hir("08.switchabsorb", all_nodes, func.name)

    def _absorb(self, func: Function, block, switch_node: SwitchNode,
                label_to_block: dict,
                _build_arm_hir, _collect_goto_targets) -> None:
        from pseudo8051.passes.ifelse import _reachable_eas, _structure_flat_ifelse
        from collections import Counter

        all_labels_here = [body for _, body in switch_node.cases if isinstance(body, str)]
        if switch_node.default_label:
            all_labels_here.append(switch_node.default_label)

        sentinel_ea = max(b.start_ea for b in func.blocks) + 1

        merge_ea = _find_switch_merge_ea(func, switch_node, block.start_ea)

        if merge_ea is None:
            # All arms truly terminate (no EA shared by 2+ arms).
            dbg("switch", "SwitchBodyAbsorber: all arms terminate — absorbing without merge")
            merge_ea = sentinel_ea
            dead_end_labels = set(all_labels_here)
        else:
            # Identify dead-end arms: those that don't reach merge_ea.
            # Use min arm EA as BFS floor (same logic as _find_switch_merge_ea).
            arm_eas_here = [label_to_block[l].start_ea
                            for l in all_labels_here if l in label_to_block]
            bfs_floor = min(arm_eas_here) - 1 if arm_eas_here else block.start_ea
            dead_end_labels = {
                label
                for label in all_labels_here
                if (blk_de := label_to_block.get(label)) is not None
                and merge_ea not in _reachable_eas(blk_de, bfs_floor)
            }
            if dead_end_labels:
                dbg("switch",
                    f"SwitchBodyAbsorber: {len(dead_end_labels)} dead-end arm(s), "
                    f"merge @ {hex(merge_ea)}")

        # Resolve the merge label (used only by live arms)
        if merge_ea == sentinel_ea:
            merge_label = ""
        else:
            merge_block = next((b for b in func.blocks if b.start_ea == merge_ea), None)
            merge_label = (_label_for(merge_block) if merge_block
                           else f"label_{hex(merge_ea).removeprefix('0x')}")
        dbg("switch", f"  SwitchBodyAbsorber: merge_ea={hex(merge_ea)} "
                      f"merge_label={merge_label!r} "
                      f"dead_end_labels={sorted(dead_end_labels)}")

        def _arm_merge(label: str) -> Tuple[int, str]:
            """Return (effective_merge_ea, effective_merge_label) for an arm."""
            if label in dead_end_labels:
                return sentinel_ea, ""
            return merge_ea, merge_label

        # Pre-compute all arm block EAs for external-reference check.
        # Use _arm_blocks_sw so cases whose entry EA > merge_ea are included.
        all_arm_eas: set = set()
        for _, label in switch_node.cases:
            if isinstance(label, str) and label in label_to_block:
                arm_me, _ = _arm_merge(label)
                for b in _arm_blocks_sw(label_to_block[label], arm_me):
                    all_arm_eas.add(b.start_ea)
        if switch_node.default_label and switch_node.default_label in label_to_block:
            arm_me, _ = _arm_merge(switch_node.default_label)
            for b in _arm_blocks_sw(label_to_block[switch_node.default_label], arm_me):
                all_arm_eas.add(b.start_ea)

        # Labels of blocks inlined into arms — gotos to these are dead after inlining.
        inlined_block_labels: set = {
            _label_for(b) for b in func.blocks if b.start_ea in all_arm_eas
        }

        # Collect external goto targets (blocks outside the switch block + arms)
        dbg("switch", f"  SwitchBodyAbsorber: all_arm_eas={sorted(hex(e) for e in all_arm_eas)}")
        external_targets: set = set()
        for blk in func.blocks:
            if (getattr(blk, "_absorbed", False)
                    or blk is block
                    or blk.start_ea in all_arm_eas):
                continue
            blk_targets = _collect_goto_targets(blk.hir)
            if blk_targets:
                dbg("switch", f"  SwitchBodyAbsorber: external block "
                              f"{hex(blk.start_ea)} goto targets: {blk_targets}")
            external_targets |= blk_targets
        dbg("switch", f"  SwitchBodyAbsorber: external_targets={external_targets}")

        all_body_blocks: List[BasicBlock] = []
        label_body_cache: dict = {}   # label → List[HIRNode]
        ext_copy_cache: dict = {}     # label → no-break HIR for external replacement
        arm_blocks_cache: dict = {}   # label → (sorted body_blocks, arm_ml) for sub-arm lookup

        def _get_body(label: str) -> Union[str, List[HIRNode]]:
            """Return inlined body HIR for label, or the label string if blocked."""
            if label in label_body_cache:
                dbg("switch", f"  _get_body({label!r}): cache hit")
                return label_body_cache[label]
            if label not in label_to_block:
                dbg("switch", f"  _get_body({label!r}): not in label_to_block — keeping goto")
                return label  # keep as goto label
            if label in external_targets:
                ext_blks = [hex(b.start_ea) for b in func.blocks
                            if not getattr(b, "_absorbed", False)
                            and b is not block
                            and b.start_ea not in all_arm_eas
                            and label in _collect_goto_targets(b.hir)]
                dbg("switch", f"  _get_body({label!r}): in external_targets "
                              f"(referenced by: {ext_blks}) — will copy to reference sites")
            arm_me, arm_ml = _arm_merge(label)
            body_blocks = _arm_blocks_sw(label_to_block[label], arm_me)
            dbg("switch", f"  _get_body({label!r}): arm_me={hex(arm_me)} arm_ml={arm_ml!r} "
                          f"body_blocks={[hex(b.start_ea) for b in body_blocks]}")
            # Keep intra-arm block-entry labels that are actually goto targets
            # within the arm so _structure_flat_ifelse can detect and structure
            # if/else diamonds.  Only keep labels that are jumped to — unused
            # block-entry labels would pollute body-text deduplication.
            arm_label_names = {_label_for(b) for b in body_blocks}
            arm_goto_targets: set = set()
            for _blk in body_blocks:
                arm_goto_targets |= _collect_goto_targets(_blk.hir)
            arm_keep_labels = arm_label_names & arm_goto_targets
            body_hir = _build_arm_hir(body_blocks, arm_ml,
                                      keep_labels=arm_keep_labels or None)
            dbg("switch", f"    after build_arm_hir: {[type(n).__name__ for n in body_hir]}")
            # Strip gotos to inlined arm blocks — these are dead fall-throughs
            # after the blocks are concatenated (e.g. goto trampoline at lower EA).
            body_hir = [n for n in body_hir
                        if not (isinstance(n, GotoStatement)
                                and n.label in inlined_block_labels)]
            # Structure flat if/else within the arm body.  SwitchBodyAbsorber runs
            # after IfElseStructurer, so arm HIR is a raw concatenation of block HIR
            # that was never through if/else structuring.  Do it now so case bodies
            # contain IfNode trees rather than bare IfGoto/GotoStatement/Label nodes.
            body_hir = _structure_flat_ifelse(body_hir)
            dbg("switch", f"    after structure_flat_ifelse: {[type(n).__name__ for n in body_hir]}")
            # Stash a no-break copy for external goto replacement before adding breaks.
            # If the body falls through to the merge, append the merge block's code so
            # the external copy includes the full execution path.
            if label in external_targets:
                import copy as _copy
                ext_body = _copy.deepcopy(body_hir)
                if arm_ml and _needs_break(ext_body):
                    merge_blk = label_to_block.get(arm_ml)
                    if merge_blk and not getattr(merge_blk, "_absorbed", False):
                        merge_hir = [n for n in merge_blk.hir
                                     if not isinstance(n, Label)]
                        ext_body = ext_body + merge_hir
                        dbg("switch", f"  _get_body: appended merge {arm_ml!r} "
                                      f"to ext copy of {label!r}")
                ext_copy_cache[label] = ext_body
            body_hir = _replace_goto_with_break(body_hir, arm_ml)
            needs_brk = _needs_break(body_hir)
            dbg("switch", f"    after replace_goto_with_break: "
                          f"{[type(n).__name__ for n in body_hir]} needs_break={needs_brk}")
            if needs_brk:
                body_hir.append(BreakStmt(switch_node.ea))
            label_body_cache[label] = body_hir
            sorted_body = sorted(body_blocks, key=lambda b: b.start_ea)
            arm_blocks_cache[label] = (sorted_body, arm_ml)
            all_body_blocks.extend(body_blocks)
            return body_hir

        # Build new cases list
        new_cases: List[Tuple[List[int], Union[str, List[HIRNode]]]] = []
        for values, body in switch_node.cases:
            if isinstance(body, str):
                new_cases.append((values, _get_body(body)))
            else:
                new_cases.append((values, body))
        switch_node.cases = new_cases

        # Merge cases whose assembled bodies are identical (e.g. multiple
        # JMP-table entries pointing at the same target block)
        deduped: List[Tuple[List[int], Union[str, List[HIRNode]]]] = []
        for values, body in new_cases:
            if isinstance(body, list):
                for ev, eb in deduped:
                    if isinstance(eb, list) and (
                            eb is body or _body_text(eb) == _body_text(body)):
                        dbg("switch", f"  dedup: merging case(s) {values} into {ev} "
                                      f"(same={'identity' if eb is body else 'text'})")
                        ev.extend(values)
                        break
                else:
                    deduped.append((list(values), body))
            else:
                deduped.append((list(values), body))
        switch_node.cases = deduped

        # Handle default body
        if switch_node.default_label and isinstance(switch_node.default_label, str):
            dbody = _get_body(switch_node.default_label)
            if isinstance(dbody, list):
                switch_node.default_body = dbody
                switch_node.default_label = None

        # Also handle external references to *intermediate* arm blocks (shared tail code
        # that sits inside a case arm but is not the case entry point).
        for case_label, (sorted_blocks, arm_ml) in arm_blocks_cache.items():
            for i, blk in enumerate(sorted_blocks):
                blk_label = _label_for(blk)
                if blk_label in external_targets and blk_label not in ext_copy_cache:
                    sub_hir = _build_arm_hir(sorted_blocks[i:], arm_ml)
                    sub_hir = [n for n in sub_hir
                               if not (isinstance(n, GotoStatement)
                                       and n.label in inlined_block_labels)]
                    # If the sub-arm falls through to the merge point, append merge
                    # block's code so the external copy has the full execution path.
                    if arm_ml and _needs_break(sub_hir):
                        merge_blk = label_to_block.get(arm_ml)
                        if merge_blk and not getattr(merge_blk, "_absorbed", False):
                            merge_hir = [n for n in merge_blk.hir
                                         if not isinstance(n, Label)]
                            sub_hir = sub_hir + merge_hir
                            dbg("switch", f"  SwitchBodyAbsorber: appended merge "
                                          f"{arm_ml!r} to ext copy of {blk_label!r}")
                    ext_copy_cache[blk_label] = sub_hir
                    dbg("switch", f"  SwitchBodyAbsorber: intermediate ext ref "
                                  f"{blk_label!r} @ {hex(blk.start_ea)} in arm "
                                  f"'{case_label}' → "
                                  f"{[type(n).__name__ for n in sub_hir]}")

        for blk in all_body_blocks:
            blk._absorbed = True

        # Replace external gotos to absorbed case labels with no-break inline copies.
        dbg("switch", f"  SwitchBodyAbsorber: ext_copy_cache keys={list(ext_copy_cache)}")
        if ext_copy_cache:
            from pseudo8051.passes.ifelse import _replace_goto_in_hir
            for ref_blk in func.blocks:
                if (getattr(ref_blk, "_absorbed", False)
                        or ref_blk is block
                        or ref_blk.start_ea in all_arm_eas):
                    continue
                for lbl, ext_hir in ext_copy_cache.items():
                    blk_targets = _collect_goto_targets(ref_blk.hir)
                    dbg("switch", f"  SwitchBodyAbsorber: ref_blk {hex(ref_blk.start_ea)} "
                                  f"targets={blk_targets} (looking for {lbl!r})")
                    if lbl in blk_targets:
                        dbg("switch", f"  SwitchBodyAbsorber: inlining {lbl!r} "
                                      f"into block {hex(ref_blk.start_ea)}")
                        ref_blk.hir = _replace_goto_in_hir(ref_blk.hir, lbl, ext_hir)
                        dbg("switch", f"  SwitchBodyAbsorber: → done, new HIR: "
                                      f"{[type(n).__name__ for n in ref_blk.hir]}")

        # Absorb the merge block if it isn't referenced by any external goto.
        # Necessary for jump-table switches where merge_ea < block.start_ea:
        # the merge block would otherwise appear before the switch in the
        # BFS-ordered output.  In the normal case (merge after switch) this is
        # also harmless — the block is absorbed in-place so nothing changes.
        if merge_ea != sentinel_ea and merge_label not in external_targets:
            merge_block = func._block_map.get(merge_ea)
            if (merge_block is not None
                    and not getattr(merge_block, "_absorbed", False)):
                merge_hir = [n for n in merge_block.hir
                             if not isinstance(n, Label)]
                sw_idx = next((i for i, n in enumerate(block.hir)
                               if n is switch_node), None)
                if sw_idx is not None and merge_hir:
                    block.hir = (block.hir[:sw_idx + 1]
                                 + merge_hir
                                 + block.hir[sw_idx + 1:])
                merge_block._absorbed = True
                dbg("switch", f"  SwitchBodyAbsorber: absorbed merge block "
                              f"{hex(merge_ea)} inline after switch")

    def _dead_label_cleanup(self, func: Function,
                            _collect_goto_targets, _drop_dead_labels) -> None:
        live: set = set()
        for blk in func.blocks:
            if not getattr(blk, "_absorbed", False):
                live |= _collect_goto_targets(blk.hir)
        for blk in func.blocks:
            if not getattr(blk, "_absorbed", False) and blk.hir:
                blk.hir = _drop_dead_labels(blk.hir, live)
