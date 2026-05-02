"""
passes/xpage_trampoline.py — CrossPageTrampolinePass

Runs after SimpleExternalInliner (when per-block HIR is flat and complete)
and before AnnotationPass.

Each "goto_X" trampoline block has the form:
    [optional Label nodes]
    Assign(Regs(DPTR), Const(offset))   ← MOV DPTR, #func_offset
    GotoStatement('*_call_page_N')      ← jump to page-switch stub

Replace with:
    [optional Label nodes]
    ExprStmt(Call(real_name, args))

Then mark the page-switch stub chain (call_page_N block → page-epilogue block)
as absorbed so they are excluded from func.hir assembly.
"""

import re
from typing import Optional

from pseudo8051.ir.hir import HIRNode, Label, Assign, GotoStatement, ExprStmt, ReturnStmt
from pseudo8051.ir.expr import Regs, Const, Call
from pseudo8051.ir.function import Function
from pseudo8051.passes import OptimizationPass
from pseudo8051.constants import dbg


def _dptr_offset(node) -> Optional[int]:
    """Return Const value if node is Assign(DPTR/DPL/DPH, Const), else None."""
    if not isinstance(node, Assign):
        return None
    if not isinstance(node.rhs, Const):
        return None
    lhs = node.lhs
    if isinstance(lhs, Regs) and set(lhs.names) <= {'DPTR', 'DPL', 'DPH'}:
        return node.rhs.value & 0xFFFF
    return None


def _build_call(ea: int, real_name: str) -> ExprStmt:
    """Build ExprStmt(Call(real_name, args)) using prototype if available."""
    try:
        from pseudo8051.prototypes import get_proto, param_regs
        from pseudo8051.ir.expr import RegGroup, Name as _Name
        _proto = get_proto(real_name)
        if _proto:
            _p_regs = param_regs(_proto)
            _args = []
            for _p, _regs in zip(_proto.params, _p_regs):
                if _regs:
                    _args.append(RegGroup(_regs) if len(_regs) > 1 else _Name("".join(_regs)))
                else:
                    _args.append(_Name(_p.name))
            return ExprStmt(ea, Call(real_name, _args))
    except Exception:
        pass
    return ExprStmt(ea, Call(real_name, []))


def _absorb_chain(label_to_block: dict, start_label: str) -> None:
    """Mark the page-switch stub chain (call_page block + epilogue block) as absorbed."""
    blk = label_to_block.get(start_label)
    if blk is None:
        return
    visited = set()
    queue = [blk]
    while queue:
        b = queue.pop(0)
        if b.start_ea in visited:
            continue
        visited.add(b.start_ea)
        b._absorbed = True
        dbg("trampolines", f"  xpage-absorb: absorbed block {hex(b.start_ea)}")
        # Follow unconditional successors that look like page-switch epilogues
        for succ in b.successors:
            if succ.start_ea not in visited:
                queue.append(succ)


class CrossPageTrampolinePass(OptimizationPass):
    """
    CFG-level cross-page trampoline detection pass.

    Runs before AnnotationPass so the trampoline blocks are absorbed before
    IfElseStructurer can pull them into IfNode bodies.
    """

    def run(self, func: Function) -> None:
        try:
            import ida_name, ida_funcs
            from pseudo8051.trampolines import _page_segment_base
        except Exception:
            return   # not running inside IDA

        from pseudo8051.passes._switch_detect import _label_for
        label_to_block = {_label_for(b): b for b in func.blocks}

        for block in func.blocks:
            if getattr(block, '_absorbed', False):
                continue

            hir = block.hir
            # Find: (optional Labels) Assign(DPTR, Const) GotoStatement(*_call_page_N)
            # Scan for the GotoStatement first, then verify the preceding Assign.
            for idx, node in enumerate(hir):
                if not isinstance(node, GotoStatement):
                    continue
                m = re.match(r'.*_call_page_(\d+)$', node.label)
                if not m:
                    continue
                page_num = int(m.group(1))

                # Find the Assign(DPTR, Const) immediately before (skipping Labels)
                j = idx - 1
                while j >= 0 and isinstance(hir[j], Label):
                    j -= 1
                if j < 0:
                    continue
                offset = _dptr_offset(hir[j])
                if offset is None:
                    continue

                # Resolve real target name
                try:
                    real_ea = _page_segment_base(page_num) | offset
                    real_name = (ida_funcs.get_func_name(real_ea)
                                 or ida_name.get_name(real_ea)
                                 or hex(real_ea))
                except Exception:
                    continue

                dbg("trampolines",
                    f"  xpage-trampoline @{hex(hir[j].ea)}: "
                    f"DPTR={hex(offset)} → page {page_num} → {real_name}")

                call_node = _build_call(hir[j].ea, real_name)
                call_node.source_nodes = [hir[j], node]

                # Replace Assign(DPTR)+GotoStatement with call; keep preceding Labels
                block.hir = hir[:j] + [call_node] + hir[idx + 1:]

                # Absorb the page-switch stub and epilogue blocks
                _absorb_chain(label_to_block, node.label)
                break  # one trampoline per block
