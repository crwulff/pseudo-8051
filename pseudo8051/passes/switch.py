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
from pseudo8051.ir.expr import Reg, Regs, Const, BinOp, Expr, UnaryOp
from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock
from pseudo8051.passes        import OptimizationPass
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
            return None
        per_arm.append(_reachable_eas(blk, bfs_floor))

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

        if changed:
            self._dead_label_cleanup(func, _collect_goto_targets, _drop_dead_labels)

    def _absorb(self, func: Function, block, switch_node: SwitchNode,
                label_to_block: dict,
                _build_arm_hir, _collect_goto_targets) -> None:
        from pseudo8051.passes.ifelse import _reachable_eas
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
            body_hir = _build_arm_hir(body_blocks, arm_ml)
            dbg("switch", f"    after build_arm_hir: {[type(n).__name__ for n in body_hir]}")
            # Strip gotos to inlined arm blocks — these are dead fall-throughs
            # after the blocks are concatenated (e.g. goto trampoline at lower EA).
            body_hir = [n for n in body_hir
                        if not (isinstance(n, GotoStatement)
                                and n.label in inlined_block_labels)]
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
