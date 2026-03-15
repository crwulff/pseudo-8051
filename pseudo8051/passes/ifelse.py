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

import re
from typing import List, Optional, TYPE_CHECKING

from pseudo8051.ir.hir    import HIRNode, Statement, IfNode, WhileNode, ForNode, Label
from pseudo8051.passes    import OptimizationPass
from pseudo8051.constants import dbg

if TYPE_CHECKING:
    from pseudo8051.ir.function   import Function
    from pseudo8051.ir.basicblock import BasicBlock


# ── Condition helpers ─────────────────────────────────────────────────────────

def _extract_condition(text: str):
    """Parse 'if (cond) goto label;' → (cond, label) or None."""
    m = re.match(r'^if \((.+)\) goto (\S+);$', text)
    return (m.group(1), m.group(2)) if m else None


def _invert_condition(cond: str) -> str:
    """Negate a C condition string, avoiding double negation."""
    if cond.startswith("!(") and cond.endswith(")"):
        return cond[2:-1]
    if cond.startswith("!"):
        return cond[1:]
    return f"!({cond})"


def _label_for(block: "BasicBlock") -> str:
    return block.label or f"label_{hex(block.start_ea).removeprefix('0x')}"


# ── BFS / arm helpers ─────────────────────────────────────────────────────────

def _reachable_eas(start: "BasicBlock", min_ea: int) -> set:
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


def _is_dead_end(block: "BasicBlock", branch_ea: int) -> bool:
    """
    True when the arm starting at block has no forward successors beyond
    branch_ea — i.e. it terminates (ret / unconditional jump out of range).
    """
    reach = _reachable_eas(block, branch_ea)
    # A dead-end arm reaches only itself (and possibly other absorbed blocks
    # that also have no forward exit).
    for ea in reach:
        blk = block._func._block_map.get(ea)
        if blk is None:
            continue
        for succ in blk.successors:
            if succ.start_ea > branch_ea and succ.start_ea not in reach:
                return False  # has at least one exit to outside the arm
    return True


def _find_merge_ea(true_block: "BasicBlock",
                   false_block: "BasicBlock",
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


def _arm_blocks(start: "BasicBlock", merge_ea: int) -> List["BasicBlock"]:
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


def _build_arm_hir(blocks: List["BasicBlock"], merge_ea: int) -> List[HIRNode]:
    """
    Concatenate HIR from all arm blocks, stripping:
      • the block's own Label node (it becomes internal to the IfNode)
      • any 'goto merge_label;' statement
      • any conditional branch whose target is the merge label
    """
    merge_label = f"label_{hex(merge_ea).removeprefix('0x')}"
    nodes: List[HIRNode] = []

    for blk in blocks:
        for node in blk.hir:
            if isinstance(node, Label):
                continue
            if isinstance(node, Statement):
                if node.text == f"goto {merge_label};":
                    continue
                parsed = _extract_condition(node.text)
                if parsed and parsed[1] == merge_label:
                    continue
            nodes.append(node)

    return nodes


# ── Dead-label cleanup ────────────────────────────────────────────────────────

def _collect_goto_targets(nodes: List[HIRNode]) -> set:
    """Recursively collect every label name referenced by a goto."""
    targets: set = set()
    for node in nodes:
        if isinstance(node, Statement):
            if node.text.startswith("goto "):
                targets.add(node.text[5:].rstrip(";"))
            else:
                parsed = _extract_condition(node.text)
                if parsed:
                    targets.add(parsed[1])
        elif isinstance(node, IfNode):
            targets |= _collect_goto_targets(node.then_nodes)
            targets |= _collect_goto_targets(node.else_nodes)
        elif isinstance(node, (WhileNode, ForNode)):
            targets |= _collect_goto_targets(node.body_nodes)
    return targets


def _drop_dead_labels(nodes: List[HIRNode], live: set) -> List[HIRNode]:
    """Remove Label nodes whose names are not in live, and their blank lines."""
    result: List[HIRNode] = []
    for node in nodes:
        if isinstance(node, Label) and node.name not in live:
            # Also drop the blank line that Label.render() emits before itself;
            # that blank line is baked into render() so nothing to strip here —
            # but if the previous item in result is a blank Statement we could
            # remove it.  Keep it simple: just skip the Label node itself.
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

    def run(self, func: "Function") -> None:
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

    def _try_structure(self, func: "Function", block: "BasicBlock") -> bool:
        succs = [s for s in block.successors
                 if not getattr(s, "_absorbed", False)]
        if len(succs) != 2:
            return False

        # Find the last conditional branch in this block's HIR
        branch_idx = -1
        cond = true_label = None
        for i, node in enumerate(block.hir):
            if isinstance(node, Statement):
                parsed = _extract_condition(node.text)
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

        dbg("ifelse", f"block {hex(block.start_ea)}: cond={cond!r} "
                      f"true={hex(true_block.start_ea)} "
                      f"false={hex(false_block.start_ea)}")

        # Find merge point via forward BFS from both arms
        merge_ea = _find_merge_ea(true_block, false_block, block.start_ea)
        if merge_ea is None:
            dbg("ifelse", f"  → no merge point found, skipping")
            return False

        dbg("ifelse", f"  merge={hex(merge_ea)}")

        # Collect blocks and build HIR for each arm
        true_arm  = _arm_blocks(true_block,  merge_ea)
        false_arm = _arm_blocks(false_block, merge_ea)

        dbg("ifelse", f"  true-arm  blocks: {[hex(b.start_ea) for b in true_arm]}")
        dbg("ifelse", f"  false-arm blocks: {[hex(b.start_ea) for b in false_arm]}")

        then_nodes = _build_arm_hir(true_arm,  merge_ea)
        else_nodes = _build_arm_hir(false_arm, merge_ea)

        # ── Canonical form: then-body should be non-empty ─────────────────
        # When the true arm jumps straight to the merge (empty then), swap
        # the arms and invert the condition so the body is always in "then".
        # Example: jnb BIT, merge  →  if (BIT) { body }  (no else)
        if not then_nodes and else_nodes:
            dbg("ifelse", f"  swapping arms (then was empty), inverted cond to "
                          f"{_invert_condition(cond)!r}")
            cond       = _invert_condition(cond)
            then_nodes = else_nodes
            else_nodes = []

        # Nothing to do if both arms are empty
        if not then_nodes and not else_nodes:
            dbg("ifelse", f"  both arms empty, skipping")
            return False

        dbg("ifelse", f"  → IfNode  cond={cond!r}  "
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
