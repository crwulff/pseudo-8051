"""
passes/ifelse.py — IfElseStructurer: conditional branches → IfNode.

Pattern:
    A ──true──► [C → C1 → C2 …] ──►
    └──false──► [B → B1 → B2 …] ──► D (merge / post-dominator)

The merge block D is found by forward BFS from both arms: it is the block
with the smallest EA that is reachable from *both* true_block and false_block.
This handles arbitrarily nested multi-block arms.

After all inner ifs are structured (reverse-EA order + repeat-until-stable),
the pass removes Label nodes that are no longer targeted by any goto.
"""

import copy
from typing import List, Optional, Tuple, Union

from pseudo8051.ir.hir      import HIRNode, IfNode, WhileNode, ForNode, DoWhileNode, Label, IfGoto, GotoStatement, SwitchNode, ExprStmt
from pseudo8051.ir.expr     import Expr, UnaryOp
from pseudo8051.passes    import OptimizationPass
from pseudo8051.constants import dbg

from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock

_Cond = Union[str, Expr]


# ── Condition helpers ─────────────────────────────────────────────────────────

def _extract_condition_node(node: HIRNode) -> Optional[Tuple[_Cond, str]]:
    """Return (condition, label) from an IfGoto node, else None."""
    if isinstance(node, IfGoto):
        return (node.cond, node.label)
    return None


def _invert_condition(cond: _Cond) -> _Cond:
    """Negate a condition (str or Expr), avoiding double negation."""
    if isinstance(cond, Expr):
        # UnaryOp("!", UnaryOp("!", x)) → x
        if isinstance(cond, UnaryOp) and cond.op == "!":
            return cond.operand
        return UnaryOp("!", cond)
    # str path
    if cond.startswith("!(") and cond.endswith(")"):
        return cond[2:-1]
    if cond.startswith("!"):
        return cond[1:]
    return f"!({cond})"


def _label_for(block: BasicBlock) -> str:
    return block.label or f"label_{hex(block.start_ea).removeprefix('0x')}"


# ── BFS / arm helpers ─────────────────────────────────────────────────────────

def _reachable_eas(start: BasicBlock, min_ea: int) -> set:
    """
    Forward BFS from start collecting all reachable block EAs.
    Only follows successors with start_ea > min_ea (stay forward in the code).
    Crosses through absorbed blocks as opaque CFG nodes so the merge point
    can be found even when an inner structured region sits in between.
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
    # A dead-end arm reaches only itself (and possibly other absorbed blocks
    # that also have no forward exit).
    for ea in reach:
        blk = block._block_map.get(ea)
        if blk is None:
            continue
        for succ in blk.successors:
            if succ.start_ea > branch_ea and succ.start_ea not in reach:
                return False  # has at least one exit to outside the arm
    return True


def _find_merge_ea(true_block: BasicBlock,
                   false_block: BasicBlock,
                   branch_ea: int) -> Optional[int]:
    """
    Return the EA of the immediate post-dominator — the block with the
    smallest EA reachable from *both* arms.

    Special case: if one arm is a dead-end (returns / jumps away with no
    forward exit), the merge is the other arm's first block — the branch
    simply wraps the dead-end arm as an if-with-exit.
    """
    reach_true  = _reachable_eas(true_block,  branch_ea)
    reach_false = _reachable_eas(false_block, branch_ea)
    common = reach_true & reach_false
    if common:
        return min(common)

    # One (or both) arms are dead-ends — check for if-with-exit pattern
    if _is_dead_end(false_block, branch_ea):
        return true_block.start_ea
    if _is_dead_end(true_block, branch_ea):
        return false_block.start_ea
    return None


def _arm_blocks(start: BasicBlock, merge_ea: int) -> List[BasicBlock]:
    """
    Collect all non-absorbed blocks belonging to one if-arm:
    reachable from start with EA < merge_ea, sorted by EA.
    Still traverses absorbed blocks so non-absorbed ones beyond them
    are not accidentally missed.
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


def _build_arm_hir(blocks: List[BasicBlock], merge_label: str) -> List[HIRNode]:
    """
    Concatenate HIR from all arm blocks, stripping:
      • the block's own Label node (it becomes internal to the IfNode)
      • any 'goto merge_label;' statement or GotoStatement to merge
      • any conditional branch whose target is the merge label
    """
    nodes: List[HIRNode] = []

    for blk in blocks:
        for node in blk.hir:
            if isinstance(node, Label):
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
    """Remove Label nodes whose names are not in live, recursing into structured bodies."""
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, Label) and node.name not in live:
            continue
        result.append(node.map_bodies(lambda ns: _drop_dead_labels(ns, live)))
    return result


def _replace_goto_in_hir(nodes: List[HIRNode], label: str,
                          replacement: List[HIRNode]) -> List[HIRNode]:
    """
    Recursively replace GotoStatement(label) with an inline copy of replacement,
    and IfGoto(cond, label) with IfNode(cond, copy_of_replacement).
    Recurses into nested IfNode, loop, and SwitchNode bodies.
    """
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, GotoStatement) and node.label == label:
            result.extend(copy.deepcopy(replacement))
        elif isinstance(node, IfGoto) and node.label == label:
            result.append(IfNode(node.ea, node.cond, copy.deepcopy(replacement)))
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


def _structure_flat_ifelse(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Apply if/else structuring to a flat HIR list containing IfGoto / GotoStatement /
    Label nodes assembled from absorbed loop-body blocks.

    Detects the diamond pattern::

        IfGoto(cond, else_label)
        [then_body]
        GotoStatement(merge_label)
        Label(else_label)+
        [else_body]
        Label(merge_label)+
        [continuation]

    and replaces it with ``IfNode(cond, then_body, else_body)`` followed by
    ``Label(merge_label)`` + continuation.

    Also detects the one-sided pattern::

        IfGoto(cond, end_label)
        [body]
        Label(end_label)+

    → ``IfNode(inverted_cond, body)  Label(end_label)``

    The ``cond`` in ``IfGoto(cond, else_label)`` is the *jump* condition: when
    cond is true the else-label arm executes, when false the fall-through executes.
    Accordingly the resulting IfNode keeps cond with the jumped-to arm as
    ``then_nodes`` and the fall-through as ``else_nodes``, matching the convention
    used by ``IfElseStructurer``.

    Recurses into existing structured node bodies via ``map_bodies``.
    Iterates until no further transformations are possible.
    """
    # Recurse into existing structured nodes before trying this level
    nodes = [node.map_bodies(_structure_flat_ifelse) for node in nodes]
    changed = True
    while changed:
        nodes, changed = _structure_flat_ifelse_pass(nodes)
    return nodes


def _structure_flat_ifelse_pass(nodes: List[HIRNode]) -> Tuple[List[HIRNode], bool]:
    """Single scan of ``_structure_flat_ifelse``. Returns (new_nodes, changed)."""
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

        # Look for a GotoStatement between IfGoto and else_label.
        # Take the LAST one — it is the end-of-then-body jump to the merge label.
        goto_pos: Optional[int] = None
        merge_label: Optional[str] = None
        for k in range(i + 1, else_pos):
            if isinstance(result[k], GotoStatement):
                goto_pos = k
                merge_label = result[k].label

        if goto_pos is not None:
            # ── Full diamond: IfGoto / then / goto(merge) / else_label / else / merge ──

            # fall-through arm: between IfGoto and GotoStatement (exclusive)
            fall_through = result[i + 1:goto_pos]

            # Find Label(merge_label) at or after else_pos
            merge_pos: Optional[int] = None
            for k in range(else_pos, len(result)):
                if isinstance(result[k], Label) and result[k].name == merge_label:
                    merge_pos = k
                    break

            if merge_pos is None:
                i += 1
                continue

            # jumped-to arm: after else_label Labels, up to merge_label
            else_start = else_pos
            while (else_start < merge_pos
                   and isinstance(result[else_start], Label)
                   and result[else_start].name == else_label):
                else_start += 1
            jumped_to = result[else_start:merge_pos]

            # Convention: then_nodes = jumped-to (cond true), else_nodes = fall-through.
            # Canonical swap: if then is empty and else isn't, invert and swap.
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

            if_node = IfNode(node.ea, eff_cond, then_nodes, else_nodes)
            if_node.ann = node.ann

            # Keep one merge label (dead-label cleanup removes it if unreferenced)
            merge_label_node = result[merge_pos]
            next_pos = merge_pos + 1
            while (next_pos < len(result)
                   and isinstance(result[next_pos], Label)
                   and result[next_pos].name == merge_label):
                next_pos += 1

            result = result[:i] + [if_node, merge_label_node] + result[next_pos:]
            return result, True   # restart to find nested patterns

        else:
            # ── One-sided: IfGoto(cond, end) / body / Label(end) ──
            body = result[i + 1:else_pos]
            body_nonempty = any(not isinstance(n, Label) for n in body)
            if body_nonempty:
                # Invert: cond-true means skip body; so body runs when cond is false
                inv_cond = _invert_condition(cond)
                if_node  = IfNode(node.ea, inv_cond, body, [])
                if_node.ann = node.ann
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
    Remove IfGoto(cond, label_X) when Label(label_X) is the immediately following
    node in the same HIR list.  Both arms of such a branch reach the same point
    so the condition is irrelevant and the branch is a no-op.
    Recurses into nested IfNode, loop, and SwitchNode bodies.
    """
    result: List[HIRNode] = []
    for i, node in enumerate(nodes):
        if isinstance(node, IfNode):
            node.then_nodes = _remove_nop_gotos(node.then_nodes)
            node.else_nodes = _remove_nop_gotos(node.else_nodes)
        elif isinstance(node, (WhileNode, ForNode, DoWhileNode)):
            node.body_nodes = _remove_nop_gotos(node.body_nodes)
        elif isinstance(node, SwitchNode):
            node.cases = [(vals, _remove_nop_gotos(body) if isinstance(body, list) else body)
                          for vals, body in node.cases]
            if isinstance(node.default_body, list):
                node.default_body = _remove_nop_gotos(node.default_body)
        if (isinstance(node, IfGoto)
                and i + 1 < len(nodes)
                and isinstance(nodes[i + 1], Label)
                and nodes[i + 1].name == node.label):
            # The branch is a nop (jumps to the immediately-following label) but the
            # condition expression may set flags (e.g. C) as a side effect, so keep
            # it as a plain expression statement rather than dropping it entirely.
            dbg("ifelse", f"  nop-goto → expr: if ({node.cond}) goto {node.label}")
            result.append(ExprStmt(node.ea, node.cond))
            continue
        result.append(node)
    return result


# ── Pass ──────────────────────────────────────────────────────────────────────

class IfElseStructurer(OptimizationPass):
    """
    Promote conditional branch blocks into IfNode HIR nodes.

    Iterates in reverse EA order (inner → outer) and repeats until stable,
    so nested if/else structures are built bottom-up correctly.
    After structuring, removes Label nodes with no remaining goto referencing
    them.
    """

    def run(self, func: Function) -> None:
        # ── Structure if/else nodes ───────────────────────────────────────
        changed = True
        passes  = 0
        while changed:
            changed = False
            passes += 1
            for block in reversed(func.blocks):
                if getattr(block, "_absorbed", False):
                    continue
                if self._try_structure(func, block):
                    changed = True
                    break   # restart — absorbed-set has changed

        dbg("ifelse", f"{passes} iteration(s) to reach fixed point")

        # ── Nop-goto removal ─────────────────────────────────────────────
        for block in func.blocks:
            if not getattr(block, "_absorbed", False) and block.hir:
                block.hir = _remove_nop_gotos(block.hir)

        # ── Dead-label cleanup ────────────────────────────────────────────
        live_labels: set = set()
        for block in func.blocks:
            if not getattr(block, "_absorbed", False):
                live_labels |= _collect_goto_targets(block.hir)

        removed: list = []
        for block in func.blocks:
            if not getattr(block, "_absorbed", False) and block.hir:
                dead = [n.name for n in block.hir
                        if isinstance(n, Label) and n.name not in live_labels]
                block.hir = _drop_dead_labels(block.hir, live_labels)
                removed.extend(dead)
        if removed:
            dbg("ifelse", f"dead labels removed: {removed}")
        from pseudo8051.passes.debug_dump import dump_pass_hir
        all_nodes = [n for b in func.blocks
                     if not getattr(b, "_absorbed", False) for n in b.hir]
        dump_pass_hir("ifelse", all_nodes, func.name)

    def _try_structure(self, func: Function, block: BasicBlock) -> bool:
        succs = [s for s in block.successors
                 if not getattr(s, "_absorbed", False)]
        if len(succs) != 2:
            return False

        # Find the last conditional branch in this block's HIR
        branch_idx = -1
        cond: Optional[_Cond] = None
        true_label: Optional[str] = None
        for i, node in enumerate(block.hir):
            parsed = _extract_condition_node(node)
            if parsed:
                branch_idx = i
                cond, true_label = parsed

        if branch_idx < 0:
            return False

        # Identify which successor is the true arm (matches the goto label)
        s0, s1 = succs
        if true_label == _label_for(s0):
            true_block, false_block = s0, s1
        elif true_label == _label_for(s1):
            true_block, false_block = s1, s0
        else:
            dbg("ifelse", f"block {hex(block.start_ea)}: label {true_label!r} "
                          f"matches neither successor "
                          f"({_label_for(s0)!r}, {_label_for(s1)!r})")
            return False

        cond_str = cond.render() if isinstance(cond, Expr) else cond
        dbg("ifelse", f"block {hex(block.start_ea)}: cond={cond_str!r} "
                      f"true={hex(true_block.start_ea)} "
                      f"false={hex(false_block.start_ea)}")

        # Find merge point via forward BFS from both arms
        merge_ea = _find_merge_ea(true_block, false_block, block.start_ea)
        if merge_ea is None:
            dbg("ifelse", f"  → no merge point found, skipping")
            return False

        dbg("ifelse", f"  merge={hex(merge_ea)}")

        # Resolve the real label for the merge block (may be an IDA symbol,
        # not the synthetic "label_XXXX" format).
        merge_block = block._block_map.get(merge_ea)
        merge_label = (_label_for(merge_block) if merge_block
                       else f"label_{hex(merge_ea).removeprefix('0x')}")

        # Collect blocks and build HIR for each arm
        true_arm  = _arm_blocks(true_block,  merge_ea)
        false_arm = _arm_blocks(false_block, merge_ea)

        dbg("ifelse", f"  true-arm  blocks: {[hex(b.start_ea) for b in true_arm]}")
        dbg("ifelse", f"  false-arm blocks: {[hex(b.start_ea) for b in false_arm]}")

        # If arm block labels are referenced by external gotos, copy the arm HIR
        # inline at each reference site so no dangling gotos remain after absorption.
        # Use _label_for() so synthetic labels (label_XXXX) are included, not just
        # explicit IDA labels stored in b.label.
        arm_labels = {_label_for(b) for b in true_arm + false_arm}
        dbg("ifelse", f"  arm_labels={arm_labels}")
        arm_eas = {b.start_ea for b in true_arm + false_arm}
        external_targets: set = set()
        for blk in func.blocks:
            if (getattr(blk, "_absorbed", False)
                    or blk is block
                    or blk.start_ea in arm_eas):
                continue
            blk_targets = _collect_goto_targets(blk.hir)
            if blk_targets:
                dbg("ifelse", f"  external block {hex(blk.start_ea)} "
                              f"goto targets: {blk_targets}")
            external_targets |= blk_targets
        dbg("ifelse", f"  external_targets={external_targets}")
        externally_ref = arm_labels & external_targets
        dbg("ifelse", f"  externally_ref={externally_ref}")
        if externally_ref:
            dbg("ifelse", f"  → arm labels {externally_ref} externally "
                          f"referenced — copying to reference sites")
            # Build sub-arm HIR for each externally-referenced label.
            # Strip intra-arm gotos; those blocks' code is included in the sub-arm.
            arm_label_to_inline: dict = {}
            for arm_list in (true_arm, false_arm):
                for i, blk in enumerate(arm_list):
                    lbl = _label_for(blk)
                    if lbl in externally_ref:
                        sub_hir = _build_arm_hir(arm_list[i:], merge_label)
                        sub_hir = [n for n in sub_hir
                                   if not (isinstance(n, GotoStatement)
                                           and n.label in arm_labels)]
                        dbg("ifelse", f"  built inline HIR for {lbl!r}: "
                                      f"{[type(n).__name__ for n in sub_hir]}")
                        arm_label_to_inline[lbl] = sub_hir
            for ref_blk in func.blocks:
                if (getattr(ref_blk, "_absorbed", False)
                        or ref_blk is block
                        or ref_blk.start_ea in arm_eas):
                    continue
                for lbl, inline_hir in arm_label_to_inline.items():
                    if lbl in _collect_goto_targets(ref_blk.hir):
                        dbg("ifelse", f"  inlining {lbl!r} into block "
                                      f"{hex(ref_blk.start_ea)}")
                        ref_blk.hir = _replace_goto_in_hir(ref_blk.hir, lbl, inline_hir)
                        dbg("ifelse", f"  → done, new HIR: "
                                      f"{[type(n).__name__ for n in ref_blk.hir]}")

        then_nodes = _build_arm_hir(true_arm,  merge_label)
        else_nodes = _build_arm_hir(false_arm, merge_label)

        # ── Canonical form: then-body should be non-empty ─────────────────
        # When the true arm jumps straight to the merge (empty then), swap
        # the arms and invert the condition so the body is always in "then".
        # Example: jnb BIT, merge  →  if (BIT) { body }  (no else)
        if not then_nodes and else_nodes:
            inv = _invert_condition(cond)
            inv_str = inv.render() if isinstance(inv, Expr) else inv
            dbg("ifelse", f"  swapping arms (then was empty), inverted cond to "
                          f"{inv_str!r}")
            cond       = inv
            then_nodes = else_nodes
            else_nodes = []

        # Nothing to do if both arms are empty
        if not then_nodes and not else_nodes:
            dbg("ifelse", f"  both arms empty, skipping")
            return False

        cond_dbg = cond.render() if isinstance(cond, Expr) else cond
        dbg("ifelse", f"  → IfNode  cond={cond_dbg!r}  "
                      f"then={len(then_nodes)} node(s)  "
                      f"else={len(else_nodes)} node(s)")

        if_node = IfNode(
            ea         = block.hir[branch_idx].ea,
            condition  = cond,
            then_nodes = then_nodes,
            else_nodes = else_nodes,
        )

        # Keep HIR before the branch, replace branch+tail with IfNode
        block.hir = block.hir[:branch_idx] + [if_node]

        # Mark arm blocks as absorbed
        for blk in true_arm + false_arm:
            blk._absorbed = True

        return True
