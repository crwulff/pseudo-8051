"""
passes/_switch_build.py — SwitchBodyAbsorber helper functions.

Contains all CFG-level and HIR-tree helpers used by SwitchBodyAbsorber to
collect arm blocks, find merge points, partition label-delimited siblings,
and absorb case bodies into SwitchNodes.
"""

import copy as _copy
from typing import Dict, List, Optional, Tuple, Union

from pseudo8051.ir.hir import (
    HIRNode, Label, IfGoto, IfNode, SwitchNode,
    GotoStatement, BreakStmt, ReturnStmt)
from pseudo8051.ir.basicblock import BasicBlock
from pseudo8051.ir.function   import Function
from pseudo8051.constants     import dbg
from pseudo8051.passes._switch_detect import _label_for


def _body_text(nodes: List[HIRNode]) -> List[str]:
    """Return flat list of rendered text lines for a case body (for deduplication)."""
    return [t for n in nodes for _, t in n.render()]


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
            continue
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

    Strategy:
    1. Full intersection — merge reachable from ALL arms: return min(common).
    2. Mode fallback — some arms are dead-ends (return / jump out): return the
       EA reachable from the *maximum* number of arms.
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
    # Exclude non-default case arm EAs from merge candidates: one arm may reach
    # another case arm's entry (e.g. via cjne fall-through), which would
    # otherwise be incorrectly chosen as the merge point.
    case_labels = [body for _, body in switch_node.cases if isinstance(body, str)]
    case_arm_ea_set = {label_to_block[l].start_ea for l in case_labels
                       if l in label_to_block}
    bfs_floor = min(arm_eas) - 1 if arm_eas else branch_ea

    per_arm: List[set] = []
    for label in all_labels:
        blk = label_to_block.get(label)
        if blk is None:
            continue
        per_arm.append(_reachable_eas(blk, bfs_floor))

    if not per_arm:
        return None

    # Strategy 1: full intersection
    common = per_arm[0].copy()
    for s in per_arm[1:]:
        common &= s
    common -= case_arm_ea_set
    if common:
        return min(common)

    # Strategy 2: mode — EA reachable from the most arms (exclude case arm entries)
    ea_count: Counter = Counter(ea for s in per_arm for ea in s
                                if ea not in case_arm_ea_set)
    max_cnt = max(ea_count.values(), default=0)
    if max_cnt < 2:
        return None
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
            node.then_nodes = _replace_goto_with_break(node.then_nodes, merge_label)
            node.else_nodes = _replace_goto_with_break(node.else_nodes, merge_label)
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


def _partition_by_switch_labels(sibling_nodes: List[HIRNode],
                                 case_labels: set):
    """
    Partition *sibling_nodes* into:
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
    sibling nodes.  When found, absorb the arms into the SwitchNode and remove
    them from the body list.
    Returns (new_nodes, changed).
    """
    result: List[HIRNode] = []
    i = 0
    changed = False
    while i < len(nodes):
        node = nodes[i]
        if isinstance(node, SwitchNode):
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

                    # Pre-structuring: absorb nested switches first.
                    for _lbl in arm_groups:
                        arm_groups[_lbl], _ = _absorb_switch_in_body_list(arm_groups[_lbl])

                    from pseudo8051.passes.ifelse import _structure_flat_ifelse
                    for _lbl in arm_groups:
                        arm_groups[_lbl] = _structure_flat_ifelse(arm_groups[_lbl])

                    # Merge-point detection
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
                                    arm_groups[_search_lbl] = _arm_hir_s[:_idx]
                                    break
                            if merge_label is not None:
                                break

                    if merge_label is not None:
                        for _lbl in arm_groups:
                            arm_groups[_lbl] = _replace_goto_with_break(
                                arm_groups[_lbl], merge_label)

                    # Embedded-tail resolution
                    embedded_tails: Dict[str, List[HIRNode]] = {}
                    for arm_nodes in arm_groups.values():
                        for idx, n in enumerate(arm_nodes):
                            if isinstance(n, Label) and n.name not in case_labels:
                                embedded_tails[n.name] = arm_nodes[idx + 1:]
                    if embedded_tails:
                        for lbl in arm_groups:
                            arm_hir = arm_groups[lbl]
                            cleaned: List[HIRNode] = []
                            for j, n in enumerate(arm_hir):
                                if (isinstance(n, GotoStatement)
                                        and n.label in embedded_tails
                                        and j + 1 < len(arm_hir)
                                        and isinstance(arm_hir[j + 1], Label)
                                        and arm_hir[j + 1].name == n.label):
                                    continue
                                cleaned.append(n)
                            arm_groups[lbl] = cleaned
                        for lbl in arm_groups:
                            arm_hir = arm_groups[lbl]
                            while (arm_hir
                                   and isinstance(arm_hir[-1], GotoStatement)
                                   and arm_hir[-1].label in embedded_tails):
                                target_label = arm_hir[-1].label
                                arm_hir.pop()
                                arm_hir.extend(embedded_tails[target_label])
                        for lbl in arm_groups:
                            arm_groups[lbl] = [
                                n for n in arm_groups[lbl]
                                if not (isinstance(n, Label)
                                        and n.name in embedded_tails)
                            ]

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
                    result.extend(post_switch_nodes)
                    i = len(nodes)
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
