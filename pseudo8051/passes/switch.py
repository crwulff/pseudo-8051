"""
passes/switch.py — SwitchStructurer and SwitchBodyAbsorber pass classes.

Detection helpers live in _switch_detect.py.
Body-absorption helpers live in _switch_build.py.
"""

from typing import List, Tuple, Union

from pseudo8051.ir.hir import (
    HIRNode, Label, SwitchNode, GotoStatement, BreakStmt)
from pseudo8051.ir.function   import Function
from pseudo8051.ir.basicblock import BasicBlock
from pseudo8051.passes        import OptimizationPass, run_blocks_until_stable, dump_hir
from pseudo8051.constants     import dbg
from pseudo8051.passes.debug_dump import dump_pass_hir

from pseudo8051.passes._switch_detect import (
    _label_for,
    _try_switch,
    _try_linear_equality_switch,
)
from pseudo8051.passes._switch_build import (
    _body_text,
    _arm_blocks_sw,
    _find_switch_merge_ea,
    _replace_goto_with_break,
    _needs_break,
    _absorb_switches_in_list,
)


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
            dbg("switch", "SwitchBodyAbsorber: all arms terminate — absorbing without merge")
            merge_ea = sentinel_ea
            dead_end_labels = set(all_labels_here)
        else:
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
            if label in dead_end_labels:
                return sentinel_ea, ""
            return merge_ea, merge_label

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

        inlined_block_labels: set = {
            _label_for(b) for b in func.blocks if b.start_ea in all_arm_eas
        }

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
        label_body_cache: dict = {}
        ext_copy_cache: dict = {}
        arm_blocks_cache: dict = {}

        def _get_body(label: str) -> Union[str, List[HIRNode]]:
            if label in label_body_cache:
                dbg("switch", f"  _get_body({label!r}): cache hit")
                return label_body_cache[label]
            if label not in label_to_block:
                dbg("switch", f"  _get_body({label!r}): not in label_to_block — keeping goto")
                return label
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
            arm_label_names = {_label_for(b) for b in body_blocks}
            arm_goto_targets: set = set()
            for _blk in body_blocks:
                arm_goto_targets |= _collect_goto_targets(_blk.hir)
            arm_keep_labels = arm_label_names & arm_goto_targets
            body_hir = _build_arm_hir(body_blocks, arm_ml,
                                      keep_labels=arm_keep_labels or None)
            dbg("switch", f"    after build_arm_hir: {[type(n).__name__ for n in body_hir]}")
            body_hir = [n for n in body_hir
                        if not (isinstance(n, GotoStatement)
                                and n.label in inlined_block_labels)]
            body_hir = _structure_flat_ifelse(body_hir)
            dbg("switch", f"    after structure_flat_ifelse: {[type(n).__name__ for n in body_hir]}")
            if label in external_targets:
                import copy as _copy
                ext_body = _copy.deepcopy(body_hir)
                if arm_ml and _needs_break(ext_body):
                    merge_blk = label_to_block.get(arm_ml)
                    if merge_blk and not getattr(merge_blk, "_absorbed", False):
                        merge_hir = [n for n in merge_blk.hir
                                     if not isinstance(n, Label)]
                        ext_body = ext_body + _copy.deepcopy(merge_hir)
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

        new_cases: List[Tuple[List[int], Union[str, List[HIRNode]]]] = []
        for values, body in switch_node.cases:
            if isinstance(body, str):
                new_cases.append((values, _get_body(body)))
            else:
                new_cases.append((values, body))
        switch_node.cases = new_cases

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

        if switch_node.default_label and isinstance(switch_node.default_label, str):
            dbody = _get_body(switch_node.default_label)
            if isinstance(dbody, list):
                switch_node.default_body = dbody
                switch_node.default_label = None

        for case_label, (sorted_blocks, arm_ml) in arm_blocks_cache.items():
            for i, blk in enumerate(sorted_blocks):
                blk_label = _label_for(blk)
                if blk_label in external_targets and blk_label not in ext_copy_cache:
                    sub_hir = _build_arm_hir(sorted_blocks[i:], arm_ml)
                    sub_hir = [n for n in sub_hir
                               if not (isinstance(n, GotoStatement)
                                       and n.label in inlined_block_labels)]
                    if arm_ml and _needs_break(sub_hir):
                        merge_blk = label_to_block.get(arm_ml)
                        if merge_blk and not getattr(merge_blk, "_absorbed", False):
                            import copy as _copy
                            merge_hir = [n for n in merge_blk.hir
                                         if not isinstance(n, Label)]
                            sub_hir = sub_hir + _copy.deepcopy(merge_hir)
                            dbg("switch", f"  SwitchBodyAbsorber: appended merge "
                                          f"{arm_ml!r} to ext copy of {blk_label!r}")
                    ext_copy_cache[blk_label] = sub_hir
                    dbg("switch", f"  SwitchBodyAbsorber: intermediate ext ref "
                                  f"{blk_label!r} @ {hex(blk.start_ea)} in arm "
                                  f"'{case_label}' → "
                                  f"{[type(n).__name__ for n in sub_hir]}")

        for blk in all_body_blocks:
            blk._absorbed = True

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
