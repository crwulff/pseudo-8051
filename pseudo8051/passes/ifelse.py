"""
passes/ifelse.py — IfElseStructurer: conditional branches → IfNode.

Helper functions (BFS, arm blocks, dead-label cleanup, flat structuring)
live in _ifelse_helpers.py and are re-exported here so existing imports
from this module continue to work.
"""

from typing import Optional, Union

from pseudo8051.ir.hir      import HIRNode, IfNode, Label, IfGoto, GotoStatement, SwitchNode
from pseudo8051.ir.expr     import Expr
from pseudo8051.passes    import BlockStructurer
from pseudo8051.constants import dbg

from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock

from pseudo8051.passes._ifelse_helpers import (  # noqa: F401 — re-exported for callers
    _invert_condition,
    _reachable_eas,
    _is_dead_end,
    _find_merge_ea,
    _arm_blocks,
    _build_arm_hir,
    _collect_goto_targets,
    _drop_dead_labels,
    _strip_gotos_to,
    _strip_redundant_exit_gotos,
    _replace_goto_in_hir,
    _structure_flat_ifelse,
    _structure_flat_ifelse_pass,
    _remove_nop_gotos,
)

_Cond = Union[str, Expr]


# ── Condition helpers ─────────────────────────────────────────────────────────

def _extract_condition_node(node: HIRNode) -> Optional[tuple]:
    """Return (condition, label) from an IfGoto node, else None."""
    if isinstance(node, IfGoto):
        return (node.cond, node.label)
    return None


def _label_for(block: BasicBlock) -> str:
    return block.label or f"label_{hex(block.start_ea).removeprefix('0x')}"


# ── Pass ──────────────────────────────────────────────────────────────────────

class IfElseStructurer(BlockStructurer):
    """
    Promote conditional branch blocks into IfNode HIR nodes.

    Iterates in reverse EA order (inner → outer) and repeats until stable.
    After structuring, removes Label nodes with no remaining goto referencing them.
    """

    block_order = "reverse"
    pass_name   = "07.ifelse"

    def try_block(self, func: Function, block: BasicBlock) -> bool:
        return self._try_structure(func, block)

    def post_run(self, func: Function) -> None:
        # ── Nop-goto removal ─────────────────────────────────────────────
        for block in func.blocks:
            if not getattr(block, "_absorbed", False) and block.hir:
                block.hir = _remove_nop_gotos(block.hir)

        # ── Strip redundant exit gotos ────────────────────────────────────
        for block in func.blocks:
            if not getattr(block, "_absorbed", False) and block.hir:
                block.hir = _strip_redundant_exit_gotos(block.hir)

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

        merge_ea = _find_merge_ea(true_block, false_block, block.start_ea)
        if merge_ea is None:
            dbg("ifelse", f"  → no merge point found, skipping")
            return False

        dbg("ifelse", f"  merge={hex(merge_ea)}")

        merge_block = block._block_map.get(merge_ea)
        merge_label = (_label_for(merge_block) if merge_block
                       else f"label_{hex(merge_ea).removeprefix('0x')}")

        true_arm  = _arm_blocks(true_block,  merge_ea)
        false_arm = _arm_blocks(false_block, merge_ea)

        dbg("ifelse", f"  true-arm  blocks: {[hex(b.start_ea) for b in true_arm]}")
        dbg("ifelse", f"  false-arm blocks: {[hex(b.start_ea) for b in false_arm]}")

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

        # Collect switch case labels and intra-arm goto targets to preserve.
        switch_labels: set = set()
        for blk in true_arm + false_arm:
            for _n in blk.hir:
                if isinstance(_n, SwitchNode):
                    for _, _body in _n.cases:
                        if isinstance(_body, str):
                            switch_labels.add(_body)
                    if isinstance(_n.default_label, str):
                        switch_labels.add(_n.default_label)

        for arm_list in (true_arm, false_arm):
            arm_label_names = {_label_for(b) for b in arm_list}
            for blk in arm_list:
                for _tgt in _collect_goto_targets(blk.hir):
                    if _tgt in arm_label_names and _tgt != merge_label:
                        switch_labels.add(_tgt)

        then_nodes = _build_arm_hir(true_arm,  merge_label,
                                    keep_labels=switch_labels or None)
        else_nodes = _build_arm_hir(false_arm, merge_label,
                                    keep_labels=switch_labels or None)

        # Canonical form: then-body should be non-empty
        if not then_nodes and else_nodes:
            inv = _invert_condition(cond)
            inv_str = inv.render() if isinstance(inv, Expr) else inv
            dbg("ifelse", f"  swapping arms (then was empty), inverted cond to "
                          f"{inv_str!r}")
            cond       = inv
            then_nodes = else_nodes
            else_nodes = []

        if not then_nodes and not else_nodes:
            dbg("ifelse", f"  both arms empty, skipping")
            return False

        cond_dbg = cond.render() if isinstance(cond, Expr) else cond
        dbg("ifelse", f"  → IfNode  cond={cond_dbg!r}  "
                      f"then={len(then_nodes)} node(s)  "
                      f"else={len(else_nodes)} node(s)")

        branch_node = block.hir[branch_idx]
        if_node = IfNode(
            ea         = branch_node.ea,
            condition  = cond,
            then_nodes = then_nodes,
            else_nodes = else_nodes,
        )
        if_node.ann          = branch_node.ann
        if_node.source_nodes = branch_node.source_nodes

        block.hir = block.hir[:branch_idx] + [if_node]

        for blk in true_arm + false_arm:
            blk._absorbed = True

        return True
