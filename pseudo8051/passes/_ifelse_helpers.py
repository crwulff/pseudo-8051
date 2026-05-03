"""
passes/_ifelse_helpers.py — IfElse structuring utility functions.

BFS/reachability helpers, arm block collection, HIR assembly, dead-label
cleanup, goto replacement, and flat if/else structuring.  All pure helper
functions with no dependency on the IfElseStructurer class.
"""

import copy
from typing import List, Optional, Tuple, Union

from pseudo8051.ir.hir import (
    HIRNode, IfNode, WhileNode, ForNode, DoWhileNode,
    Label, IfGoto, GotoStatement, SwitchNode, ExprStmt)
from pseudo8051.ir.expr     import Expr, UnaryOp
from pseudo8051.ir.basicblock import BasicBlock
from pseudo8051.constants import dbg

_Cond = Union[str, Expr]


# ── Condition helpers ─────────────────────────────────────────────────────────

def _invert_condition(cond: _Cond) -> _Cond:
    """Negate a condition (str or Expr), avoiding double negation."""
    if isinstance(cond, Expr):
        if isinstance(cond, UnaryOp) and cond.op == "!":
            return cond.operand
        return UnaryOp("!", cond)
    if cond.startswith("!(") and cond.endswith(")"):
        return cond[2:-1]
    if cond.startswith("!"):
        return cond[1:]
    return f"!({cond})"


# ── BFS / arm helpers ─────────────────────────────────────────────────────────

def _reachable_eas(start: BasicBlock, min_ea: int) -> set:
    """
    Forward BFS from start collecting all reachable block EAs.
    Only follows successors with start_ea > min_ea.
    Crosses through absorbed blocks for CFG connectivity.
    """
    visited: set = set()
    queue = [start]
    while queue:
        blk = queue.pop(0)
        if blk.start_ea in visited:
            continue
        visited.add(blk.start_ea)
        for succ in blk.successors:
            if succ.start_ea > min_ea and succ.start_ea not in visited:
                queue.append(succ)
    return visited


def _is_dead_end(block: BasicBlock, branch_ea: int) -> bool:
    """
    True when the arm starting at block has no forward successors beyond
    branch_ea — i.e. it terminates (ret / unconditional jump out of range).
    """
    reach = _reachable_eas(block, branch_ea)
    for ea in reach:
        blk = block._block_map.get(ea)
        if blk is None:
            continue
        for succ in blk.successors:
            if succ.start_ea > branch_ea and succ.start_ea not in reach:
                return False
    return True


def _find_merge_ea(true_block: BasicBlock,
                   false_block: BasicBlock,
                   branch_ea: int) -> Optional[int]:
    """
    Return the EA of the immediate post-dominator — the block with the
    smallest EA reachable from *both* arms.

    Special case: if one arm is a dead-end, the merge is the other arm's
    first block.
    """
    reach_true  = _reachable_eas(true_block,  branch_ea)
    reach_false = _reachable_eas(false_block, branch_ea)
    common = reach_true & reach_false
    if common:
        return min(common)
    if _is_dead_end(false_block, branch_ea):
        return true_block.start_ea
    if _is_dead_end(true_block, branch_ea):
        return false_block.start_ea
    return None


def _arm_blocks(start: BasicBlock, merge_ea: int) -> List[BasicBlock]:
    """
    Collect all non-absorbed blocks belonging to one if-arm:
    reachable from start with EA < merge_ea, sorted by EA.
    """
    visited: set = set()
    result:  list = []
    queue = [start]
    while queue:
        blk = queue.pop(0)
        if blk.start_ea in visited:
            continue
        visited.add(blk.start_ea)
        if blk.start_ea < merge_ea:
            if not getattr(blk, "_absorbed", False):
                result.append(blk)
            for succ in blk.successors:
                if succ.start_ea not in visited and succ.start_ea < merge_ea:
                    queue.append(succ)
    return sorted(result, key=lambda b: b.start_ea)


def _build_arm_hir(blocks: List[BasicBlock], merge_label: str,
                   keep_labels: Optional[set] = None) -> List[HIRNode]:
    """
    Concatenate HIR from all arm blocks, stripping:
      • the block's own Label node (it becomes internal to the IfNode)
      • any 'goto merge_label;' or GotoStatement to merge
      • any conditional branch whose target is the merge label

    If *keep_labels* is provided, Label nodes in that set are preserved.
    """
    nodes: List[HIRNode] = []
    for blk in blocks:
        for node in blk.hir:
            if isinstance(node, Label):
                if keep_labels and node.name in keep_labels:
                    nodes.append(node)
                continue
            if isinstance(node, GotoStatement) and node.label == merge_label:
                continue
            if isinstance(node, IfGoto) and node.label == merge_label:
                continue
            nodes.append(node)
    return nodes


# ── Dead-label cleanup ────────────────────────────────────────────────────────

def _collect_goto_targets(nodes: List[HIRNode]) -> set:
    """Recursively collect every label name referenced by a goto."""
    targets: set = set()
    for node in nodes:
        if isinstance(node, GotoStatement):
            targets.add(node.label)
        elif isinstance(node, IfGoto):
            targets.add(node.label)
        elif isinstance(node, IfNode):
            targets |= _collect_goto_targets(node.then_nodes)
            targets |= _collect_goto_targets(node.else_nodes)
        elif isinstance(node, (WhileNode, ForNode, DoWhileNode)):
            targets |= _collect_goto_targets(node.body_nodes)
        elif isinstance(node, SwitchNode):
            for _, body in node.cases:
                if isinstance(body, str):
                    targets.add(body)
                else:
                    targets |= _collect_goto_targets(body)
            if node.default_label is not None:
                targets.add(node.default_label)
            if node.default_body is not None:
                targets |= _collect_goto_targets(node.default_body)
    return targets


def _drop_dead_labels(nodes: List[HIRNode], live: set) -> List[HIRNode]:
    """Remove Label nodes not in live, recursing into structured bodies."""
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, Label) and node.name not in live:
            continue
        result.append(node.map_bodies(lambda ns: _drop_dead_labels(ns, live)))
    return result


def _strip_gotos_to(nodes: List[HIRNode], labels: set) -> List[HIRNode]:
    """
    Strip GotoStatement(X) and IfGoto(_, X) where X is in *labels* from *nodes*,
    recursing into IfNode then/else bodies.
    """
    result = []
    for node in nodes:
        if isinstance(node, GotoStatement) and node.label in labels:
            continue
        if isinstance(node, IfGoto) and node.label in labels:
            continue
        if isinstance(node, IfNode):
            node.then_nodes = _strip_gotos_to(node.then_nodes, labels)
            node.else_nodes = _strip_gotos_to(node.else_nodes, labels)
        result.append(node)
    return result


def _strip_redundant_exit_gotos(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    For each IfNode immediately followed by Label(s) in the same list, remove
    GotoStatement(X) / IfGoto(_, X) from within the IfNode bodies where X is
    one of those following labels.
    Recurses bottom-up; iterates until stable.
    """
    result = [node.map_bodies(_strip_redundant_exit_gotos) for node in nodes]
    changed = True
    while changed:
        changed = False
        for i, node in enumerate(result):
            if not isinstance(node, IfNode):
                continue
            following_labels: set = set()
            j = i + 1
            while j < len(result) and isinstance(result[j], Label):
                following_labels.add(result[j].name)
                j += 1
            if not following_labels:
                continue
            new_then = _strip_gotos_to(node.then_nodes, following_labels)
            new_else = _strip_gotos_to(node.else_nodes, following_labels)
            if new_then != node.then_nodes or new_else != node.else_nodes:
                node.then_nodes = new_then
                node.else_nodes = new_else
                changed = True
    return result


def _replace_goto_in_hir(nodes: List[HIRNode], label: str,
                          replacement: List[HIRNode]) -> List[HIRNode]:
    """
    Recursively replace GotoStatement(label) with an inline copy of replacement,
    and IfGoto(cond, label) with IfNode(cond, copy_of_replacement).
    """
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, GotoStatement) and node.label == label:
            result.extend(copy.deepcopy(replacement))
        elif isinstance(node, IfGoto) and node.label == label:
            result.append(node.copy_meta_to(IfNode(node.ea, node.cond, copy.deepcopy(replacement))))
        elif isinstance(node, IfNode):
            node.then_nodes = _replace_goto_in_hir(node.then_nodes, label, replacement)
            node.else_nodes = _replace_goto_in_hir(node.else_nodes, label, replacement)
            result.append(node)
        elif isinstance(node, (WhileNode, ForNode, DoWhileNode)):
            node.body_nodes = _replace_goto_in_hir(node.body_nodes, label, replacement)
            result.append(node)
        elif isinstance(node, SwitchNode):
            new_cases = []
            for vals, body in node.cases:
                if isinstance(body, list):
                    body = _replace_goto_in_hir(body, label, replacement)
                new_cases.append((vals, body))
            node.cases = new_cases
            if isinstance(node.default_body, list):
                node.default_body = _replace_goto_in_hir(node.default_body, label, replacement)
            result.append(node)
        else:
            result.append(node)
    return result


# ── Flat if/else structuring ──────────────────────────────────────────────────

def _count_goto_refs(nodes: List[HIRNode], label: str) -> int:
    """Count total GotoStatement references to label anywhere in the HIR tree."""
    count = 0
    for node in nodes:
        if isinstance(node, GotoStatement) and node.label == label:
            count += 1
        elif isinstance(node, IfNode):
            count += _count_goto_refs(node.then_nodes, label)
            count += _count_goto_refs(node.else_nodes, label)
        elif isinstance(node, (WhileNode, ForNode, DoWhileNode)):
            count += _count_goto_refs(node.body_nodes, label)
        elif isinstance(node, SwitchNode):
            for _, body in node.cases:
                if isinstance(body, list):
                    count += _count_goto_refs(body, label)
            if isinstance(node.default_body, list):
                count += _count_goto_refs(node.default_body, label)
    return count


def _replace_goto_with_nodes(nodes: List[HIRNode], label: str,
                              replacement: List[HIRNode]) -> Tuple[List[HIRNode], bool]:
    """Replace every GotoStatement(label) with a copy of replacement, recursing into bodies."""
    result: List[HIRNode] = []
    changed = False
    for node in nodes:
        if isinstance(node, GotoStatement) and node.label == label:
            result.extend(copy.deepcopy(replacement))
            changed = True
        elif isinstance(node, IfNode):
            new_then, c1 = _replace_goto_with_nodes(node.then_nodes, label, replacement)
            new_else, c2 = _replace_goto_with_nodes(node.else_nodes, label, replacement)
            if c1 or c2:
                result.append(node.copy_meta_to(
                    IfNode(node.ea, node.condition, new_then, new_else)))
                changed = True
            else:
                result.append(node)
        elif isinstance(node, (WhileNode, ForNode, DoWhileNode)):
            new_body, c = _replace_goto_with_nodes(node.body_nodes, label, replacement)
            if c:
                import copy as _copy
                new_node = _copy.copy(node)
                new_node.body_nodes = new_body
                result.append(new_node)
                changed = True
            else:
                result.append(node)
        elif isinstance(node, SwitchNode):
            new_cases = []
            sc = False
            for case_val, body in node.cases:
                if isinstance(body, list):
                    new_body, c = _replace_goto_with_nodes(body, label, replacement)
                    new_cases.append((case_val, new_body))
                    sc = sc or c
                else:
                    new_cases.append((case_val, body))
            new_default = node.default_body
            if isinstance(node.default_body, list):
                new_default, c = _replace_goto_with_nodes(node.default_body, label, replacement)
                sc = sc or c
            if sc:
                import copy as _copy
                new_node = _copy.copy(node)
                new_node.cases = new_cases
                new_node.default_body = new_default
                result.append(new_node)
                changed = True
            else:
                result.append(node)
        else:
            result.append(node)
    return result, changed


def _inline_singleton_goto_targets(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Find flat-level labels referenced by exactly one GotoStatement anywhere in
    the HIR tree (including inside structured bodies).  For each such label,
    inline its code block at the goto site and remove the flat-level label block.

    This handles the case where _structure_flat_ifelse has already structured
    the goto into an IfNode body, making the goto invisible to the flat-level
    structurer that matches Label nodes.
    """
    changed = True
    while changed:
        changed = False
        # Find all flat-level Labels and collect the code block after each.
        # A "singleton label block" runs from the Label to the next Label or end.
        flat_labels: dict = {}  # label_name → (label_idx, end_idx)
        for i, node in enumerate(nodes):
            if isinstance(node, Label):
                flat_labels[node.name] = i

        for lbl_name, lbl_idx in flat_labels.items():
            # Count total goto references across the whole tree
            if _count_goto_refs(nodes, lbl_name) != 1:
                continue
            # The label must not be targeted by any IfGoto (only GotoStatement)
            ifgoto_targets = set()
            for nd in nodes:
                if isinstance(nd, IfGoto):
                    ifgoto_targets.add(nd.label)
            if lbl_name in ifgoto_targets:
                continue

            # Find the extent of the label's code block (up to the next Label or end).
            end_idx = lbl_idx + 1
            while end_idx < len(nodes) and not isinstance(nodes[end_idx], Label):
                end_idx += 1

            # The body to inline is everything after the Label node (not the label itself).
            body = nodes[lbl_idx + 1:end_idx]

            # Replace the single goto with the body, then remove the flat-level block.
            new_nodes, did_replace = _replace_goto_with_nodes(nodes, lbl_name, body)
            if did_replace:
                # Remove the flat-level Label + body (now inlined at the goto site).
                new_nodes = new_nodes[:lbl_idx] + new_nodes[end_idx:]
                nodes = new_nodes
                changed = True
                break  # restart after any change

    return nodes


def _structure_flat_ifelse(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Apply if/else structuring to a flat HIR list containing IfGoto / GotoStatement /
    Label nodes assembled from absorbed loop-body blocks.

    Detects the diamond pattern and one-sided pattern; see _structure_flat_ifelse_pass
    for details.  Recurses into existing structured node bodies via map_bodies.
    Iterates until stable.
    """
    nodes = [node.map_bodies(_structure_flat_ifelse) for node in nodes]
    changed = True
    while changed:
        nodes, changed = _structure_flat_ifelse_pass(nodes)
    nodes = _inline_singleton_goto_targets(nodes)
    nodes = _strip_redundant_exit_gotos(nodes)
    return nodes


def _structure_flat_ifelse_pass(nodes: List[HIRNode]) -> Tuple[List[HIRNode], bool]:
    """Single scan of _structure_flat_ifelse. Returns (new_nodes, changed)."""
    result = list(nodes)
    i = 0
    while i < len(result):
        node = result[i]
        if not isinstance(node, IfGoto):
            i += 1
            continue

        cond       = node.cond
        else_label = node.label   # IfGoto jumps here when condition is TRUE

        # Find the first Label(else_label) after position i.
        else_pos: Optional[int] = None
        for k in range(i + 1, len(result)):
            if isinstance(result[k], Label) and result[k].name == else_label:
                else_pos = k
                break

        if else_pos is None:
            i += 1
            continue

        # Look for a GotoStatement between IfGoto and else_label (last one = end-of-then).
        goto_pos: Optional[int] = None
        merge_label: Optional[str] = None
        for k in range(i + 1, else_pos):
            if isinstance(result[k], GotoStatement):
                goto_pos = k
                merge_label = result[k].label

        if goto_pos is not None:
            # ── Full diamond: IfGoto / then / goto(merge) / else_label / else / merge ──
            fall_through = result[i + 1:goto_pos]

            merge_pos: Optional[int] = None
            for k in range(else_pos, len(result)):
                if isinstance(result[k], Label) and result[k].name == merge_label:
                    merge_pos = k
                    break

            if merge_pos is None:
                else_start = else_pos
                while (else_start < len(result)
                       and isinstance(result[else_start], Label)
                       and result[else_start].name == else_label):
                    else_start += 1
                else_end = else_start
                while else_end < len(result) and not isinstance(result[else_end], Label):
                    else_end += 1
                else_arm_raw = result[else_start:else_end]
                if (else_arm_raw
                        and isinstance(else_arm_raw[-1], GotoStatement)
                        and else_arm_raw[-1].label == merge_label):
                    # Both arms exit to the same external label.
                    jumped_to_ext    = else_arm_raw[:-1]
                    fall_through_ext = result[i + 1:goto_pos]
                    then_nodes = jumped_to_ext
                    else_nodes = fall_through_ext
                    eff_cond   = cond
                    then_nonempty = any(not isinstance(n, Label) for n in then_nodes)
                    else_nonempty = any(not isinstance(n, Label) for n in else_nodes)
                    if not then_nonempty and else_nonempty:
                        eff_cond   = _invert_condition(cond)
                        then_nodes = else_nodes
                        else_nodes = []
                    if not then_nonempty and not else_nonempty:
                        i += 1
                        continue
                    if_node = node.copy_meta_to(IfNode(node.ea, eff_cond, then_nodes, else_nodes))
                    goto_node = result[goto_pos]
                    result = result[:i] + [if_node, goto_node] + result[else_end:]
                    return result, True

                # Asymmetric diamond: fall-through exits explicitly; jumped-to arm
                # is at end of list and falls through implicitly (no terminal goto).
                if (else_end == len(result)
                        and else_arm_raw
                        and not isinstance(else_arm_raw[-1], GotoStatement)):
                    fall_through_no_goto = result[i + 1:goto_pos]
                    then_nodes = else_arm_raw
                    else_nodes = fall_through_no_goto
                    eff_cond   = cond
                    then_nonempty = any(not isinstance(n, Label) for n in then_nodes)
                    else_nonempty = any(not isinstance(n, Label) for n in else_nodes)
                    if not then_nonempty and else_nonempty:
                        eff_cond   = _invert_condition(cond)
                        then_nodes = else_nodes
                        else_nodes = []
                    if not then_nonempty and not else_nonempty:
                        i += 1
                        continue
                    if_node = node.copy_meta_to(IfNode(node.ea, eff_cond, then_nodes, else_nodes))
                    goto_node = result[goto_pos]
                    result = result[:i] + [if_node, goto_node]
                    return result, True

                i += 1
                continue

            # Full diamond with internal merge label
            else_start = else_pos
            while (else_start < merge_pos
                   and isinstance(result[else_start], Label)
                   and result[else_start].name == else_label):
                else_start += 1
            jumped_to = result[else_start:merge_pos]

            then_nodes = jumped_to
            else_nodes = fall_through
            eff_cond   = cond
            then_nonempty = any(not isinstance(n, Label) for n in then_nodes)
            else_nonempty = any(not isinstance(n, Label) for n in else_nodes)
            if not then_nonempty and else_nonempty:
                eff_cond   = _invert_condition(cond)
                then_nodes = else_nodes
                else_nodes = []

            if not then_nonempty and not else_nonempty:
                i += 1
                continue

            if_node = node.copy_meta_to(IfNode(node.ea, eff_cond, then_nodes, else_nodes))
            merge_label_node = result[merge_pos]
            next_pos = merge_pos + 1
            while (next_pos < len(result)
                   and isinstance(result[next_pos], Label)
                   and result[next_pos].name == merge_label):
                next_pos += 1
            result = result[:i] + [if_node, merge_label_node] + result[next_pos:]
            return result, True

        else:
            # ── One-sided: IfGoto(cond, end) / body / Label(end) ──
            body = result[i + 1:else_pos]
            body_nonempty = any(not isinstance(n, Label) for n in body)
            if body_nonempty:
                inv_cond = _invert_condition(cond)
                if_node  = node.copy_meta_to(IfNode(node.ea, inv_cond, body, []))
                end_label_node = result[else_pos]
                next_pos = else_pos + 1
                while (next_pos < len(result)
                       and isinstance(result[next_pos], Label)
                       and result[next_pos].name == else_label):
                    next_pos += 1
                result = result[:i] + [if_node, end_label_node] + result[next_pos:]
                return result, True

        i += 1

    return result, False


def _remove_nop_gotos(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Remove IfGoto(cond, label_X) when Label(label_X) immediately follows in
    the same HIR list — both arms reach the same point so the branch is a no-op.
    Keeps the condition as a plain ExprStmt in case it has side-effects.
    """
    result: List[HIRNode] = []
    for i, node in enumerate(nodes):
        node = node.map_bodies(_remove_nop_gotos)
        if (isinstance(node, IfGoto)
                and i + 1 < len(nodes)
                and isinstance(nodes[i + 1], Label)
                and nodes[i + 1].name == node.label):
            dbg("ifelse", f"  nop-goto → expr: if ({node.cond}) goto {node.label}")
            expr_stmt = ExprStmt(node.ea, node.cond)
            expr_stmt.source_nodes = [node]
            result.append(expr_stmt)
            continue
        result.append(node)
    return result
