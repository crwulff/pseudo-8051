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
    HIRNode, Label, Assign, CompoundAssign, IfGoto, SwitchNode,
    GotoStatement, BreakStmt, IfNode, WhileNode, ForNode, DoWhileNode, ReturnStmt)
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

    cases = [(sorted(cases_dict[lbl]), lbl) for lbl in cases_order]
    cases.sort(key=lambda pair: min(pair[0]))

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


# ── SwitchBodyAbsorber helpers ────────────────────────────────────────────────

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
    Find the merge point (post-dominator) for a SwitchNode by forward BFS from
    every case target block and the default.  Returns the smallest common EA,
    or None if no common block is reachable from all arms.
    """
    label_to_block = {_label_for(b): b for b in func.blocks}
    all_labels: List[str] = [body for _, body in switch_node.cases
                              if isinstance(body, str)]
    if switch_node.default_label:
        all_labels.append(switch_node.default_label)
    return _merge_ea_from_labels(all_labels, label_to_block, branch_ea)


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


class SwitchBodyAbsorber(OptimizationPass):
    """
    Absorb case body blocks directly into their SwitchNode.

    Must run after IfElseStructurer so case bodies are already fully structured
    (nested if/else built) before absorption.
    """

    def run(self, func: Function) -> None:
        from pseudo8051.passes.ifelse import (
            _arm_blocks, _build_arm_hir,
            _collect_goto_targets, _drop_dead_labels,
        )

        label_to_block = {_label_for(b): b for b in func.blocks}
        changed = False

        for block in func.blocks:
            if getattr(block, "_absorbed", False):
                continue
            for node in block.hir:
                if not isinstance(node, SwitchNode):
                    continue
                if not any(isinstance(body, str) for _, body in node.cases):
                    continue  # already absorbed
                self._absorb(func, block, node, label_to_block,
                             _arm_blocks, _build_arm_hir, _collect_goto_targets)
                changed = True

        if changed:
            self._dead_label_cleanup(func, _collect_goto_targets, _drop_dead_labels)

    def _absorb(self, func: Function, block, switch_node: SwitchNode,
                label_to_block: dict,
                _arm_blocks, _build_arm_hir, _collect_goto_targets) -> None:
        from pseudo8051.passes.ifelse import _reachable_eas
        from collections import Counter

        all_labels_here = [body for _, body in switch_node.cases if isinstance(body, str)]
        if switch_node.default_label:
            all_labels_here.append(switch_node.default_label)

        sentinel_ea = max(b.start_ea for b in func.blocks) + 1

        merge_ea = _find_switch_merge_ea(func, switch_node, block.start_ea)

        if merge_ea is None:
            # Find the merge EA by looking for any EA reachable from 2+ arms.
            # Arms that reach the merge EA are "live" (they goto merge → break);
            # arms that don't are "dead-end" (they terminate directly → return).
            # This correctly handles the case where the merge block itself is a
            # dead-end (e.g. it ends with ret) — `_is_dead_end` would wrongly
            # classify live arms as dead in that situation.
            arm_reach: dict = {}
            for label in all_labels_here:
                blk = label_to_block.get(label)
                if blk is not None:
                    arm_reach[label] = _reachable_eas(blk, block.start_ea)

            ea_count: Counter = Counter()
            for reach in arm_reach.values():
                for ea in reach:
                    ea_count[ea] += 1
            shared = {ea for ea, cnt in ea_count.items() if cnt >= 2}

            if shared:
                merge_ea = min(shared)
                dead_end_labels = {label for label, reach in arm_reach.items()
                                   if merge_ea not in reach}
                dbg("switch",
                    f"SwitchBodyAbsorber: mixed arms — {len(dead_end_labels)} dead-end, "
                    f"{len(all_labels_here) - len(dead_end_labels)} live")
            else:
                # No shared successors — all arms truly terminate
                dbg("switch", "SwitchBodyAbsorber: all arms terminate — absorbing without merge")
                merge_ea = sentinel_ea
                dead_end_labels = set(all_labels_here)
        else:
            dead_end_labels: set = set()

        # Resolve the merge label (used only by live arms)
        if merge_ea == sentinel_ea:
            merge_label = ""
        else:
            merge_block = next((b for b in func.blocks if b.start_ea == merge_ea), None)
            merge_label = (_label_for(merge_block) if merge_block
                           else f"label_{hex(merge_ea).removeprefix('0x')}")

        def _arm_merge(label: str) -> Tuple[int, str]:
            """Return (effective_merge_ea, effective_merge_label) for an arm."""
            if label in dead_end_labels:
                return sentinel_ea, ""
            return merge_ea, merge_label

        # Pre-compute all arm block EAs for external-reference check
        all_arm_eas: set = set()
        for _, label in switch_node.cases:
            if isinstance(label, str) and label in label_to_block:
                arm_me, _ = _arm_merge(label)
                for b in _arm_blocks(label_to_block[label], arm_me):
                    all_arm_eas.add(b.start_ea)
        if switch_node.default_label and switch_node.default_label in label_to_block:
            arm_me, _ = _arm_merge(switch_node.default_label)
            for b in _arm_blocks(label_to_block[switch_node.default_label], arm_me):
                all_arm_eas.add(b.start_ea)

        # Collect external goto targets (blocks outside the switch block + arms)
        dbg("switch", f"  SwitchBodyAbsorber: all_arm_eas={sorted(hex(e) for e in all_arm_eas)}")
        external_targets: set = set()
        for blk in func.blocks:
            if (getattr(blk, "_absorbed", False)
                    or blk is block
                    or blk.start_ea in all_arm_eas):
                continue
            external_targets |= _collect_goto_targets(blk.hir)

        all_body_blocks: List[BasicBlock] = []
        label_body_cache: dict = {}  # label → List[HIRNode]

        def _get_body(label: str) -> Union[str, List[HIRNode]]:
            """Return inlined body HIR for label, or the label string if blocked."""
            if label in label_body_cache:
                return label_body_cache[label]
            if label not in label_to_block:
                dbg("switch", f"  SwitchBodyAbsorber: {label!r} not in label_to_block — keeping goto")
                return label  # keep as goto label
            if label in external_targets:
                ext_blks = [hex(b.start_ea) for b in func.blocks
                            if not getattr(b, "_absorbed", False)
                            and b is not block
                            and b.start_ea not in all_arm_eas
                            and label in _collect_goto_targets(b.hir)]
                dbg("switch", f"  SwitchBodyAbsorber: {label!r} in external_targets "
                              f"(referenced by blocks: {ext_blks}) — keeping goto")
                return label  # keep as goto label
            arm_me, arm_ml = _arm_merge(label)
            body_blocks = _arm_blocks(label_to_block[label], arm_me)
            body_hir = _build_arm_hir(body_blocks, arm_ml)
            body_hir = _replace_goto_with_break(body_hir, arm_ml)
            if _needs_break(body_hir):
                body_hir.append(BreakStmt(switch_node.ea))
            label_body_cache[label] = body_hir
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

        # Handle default body
        if switch_node.default_label and isinstance(switch_node.default_label, str):
            dbody = _get_body(switch_node.default_label)
            if isinstance(dbody, list):
                switch_node.default_body = dbody
                switch_node.default_label = None

        for blk in all_body_blocks:
            blk._absorbed = True

    def _dead_label_cleanup(self, func: Function,
                            _collect_goto_targets, _drop_dead_labels) -> None:
        live: set = set()
        for blk in func.blocks:
            if not getattr(blk, "_absorbed", False):
                live |= _collect_goto_targets(blk.hir)
        for blk in func.blocks:
            if not getattr(blk, "_absorbed", False) and blk.hir:
                blk.hir = _drop_dead_labels(blk.hir, live)
