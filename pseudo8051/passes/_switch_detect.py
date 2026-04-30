"""
passes/_switch_detect.py — Switch pattern detection helpers.

Contains label helpers, A-delta extraction, single-step detectors, and the
two top-level detection functions (_try_switch, _try_linear_equality_switch)
that build SwitchNodes from basic-block chains.
"""

from typing import Dict, List, Optional, Tuple

from pseudo8051.ir.hir import (
    Label, Assign, CompoundAssign, ExprStmt, IfGoto, SwitchNode)
from pseudo8051.ir.expr import Reg, Regs, Const, BinOp, Expr, UnaryOp
from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock
from pseudo8051.constants     import dbg
from pseudo8051.passes.patterns._utils import _contains_a


def _label_for(block: BasicBlock) -> str:
    return block.label or f"label_{hex(block.start_ea).removeprefix('0x')}"


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
    _step_eas_by_label: Dict[str, frozenset] = {}
    _default_step_eas: frozenset = frozenset()
    _all_step_nodes: list = []

    for delta, label, is_ne, blk in steps:
        _hir_nl = [n for n in blk.hir if not isinstance(n, Label)]
        _tl = 2
        if (len(_hir_nl) >= 3
                and isinstance(_hir_nl[-3], Assign)
                and _hir_nl[-3].lhs == Reg("A")
                and not _contains_a(_hir_nl[-3].rhs)):
            _tl = 3
        _step_nodes = _hir_nl[-_tl:]
        _step_ea_set = frozenset().union(*(n.src_eas for n in _step_nodes))
        _all_step_nodes.extend(_step_nodes)
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
    if _all_step_nodes:
        sw.source_nodes = _all_step_nodes

    # Keep any Label nodes and all preamble code that precedes the switch tail.
    hir_no_labels = [n for n in head_block.hir if not isinstance(n, Label)]
    tail_len = 2
    if (len(hir_no_labels) >= 3
            and isinstance(hir_no_labels[-3], Assign)
            and hir_no_labels[-3].lhs == Reg("A")
            and not _contains_a(hir_no_labels[-3].rhs)):
        tail_len = 3
    preamble   = hir_no_labels[:-tail_len]
    label_nodes = [n for n in head_block.hir if isinstance(n, Label)]
    head_block.hir = label_nodes + preamble + [sw]

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

    Returns True if a SwitchNode was created and intermediate blocks absorbed.
    """
    steps: List[Tuple] = []   # (case_val, label, is_ne, block)
    subjects: List[Expr] = []
    cur = head_block

    # current_T tracks the net transform T applied to the subject since last reload.
    current_T = ('add', 0)

    while True:
        result = _extract_linear_equality_step(cur)
        if result is None:
            break
        transform, label, is_ne, subject, did_reload = result

        if did_reload:
            current_T = ('add', 0)

        t_type, t_val = transform

        if current_T[0] == 'add' and t_type == 'add':
            new_T = ('add', (current_T[1] + t_val) & 0xFF)
        elif current_T[0] == 'add' and current_T[1] == 0 and t_type == 'xor':
            new_T = ('xor', t_val)
        elif current_T[0] == 'xor' and t_type == 'xor':
            new_T = ('xor', current_T[1] ^ t_val)
        else:
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
    _all_step_nodes: list = []

    for case_val, label, is_ne, blk, dr in steps:
        _hir_nl = [n for n in blk.hir if not isinstance(n, Label)]
        _tl = 3 if dr else 2
        _step_nodes = _hir_nl[-_tl:]
        _step_ea_set = frozenset().union(*(n.src_eas for n in _step_nodes))
        _all_step_nodes.extend(_step_nodes)
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
    if _all_step_nodes:
        sw.source_nodes = _all_step_nodes

    # Keep Label nodes and any preamble before the switch tail in head block.
    hir_no_labels = [n for n in head_block.hir if not isinstance(n, Label)]
    first_step_result = _extract_linear_equality_step(head_block)
    tail_len = 2 + (1 if first_step_result[4] else 0)
    preamble = hir_no_labels[:-tail_len]
    label_nodes = [n for n in head_block.hir if isinstance(n, Label)]
    head_block.hir = label_nodes + preamble + [sw]

    for _, _, _, blk, _ in steps[1:]:
        blk._absorbed = True

    return True
