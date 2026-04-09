"""
passes/typesimplify/_pass.py — TypeAwareSimplifier pass class.
"""

from pseudo8051.ir.hir    import VarDecl
from pseudo8051.passes    import OptimizationPass
from pseudo8051.passes.patterns.mb_assign import collapse_mb_assigns
from pseudo8051.passes.patterns._utils  import VarInfo
from pseudo8051.ir.function import Function
from pseudo8051.constants import PARAM_REG_ORDER

from pseudo8051.passes.typesimplify._regmap   import (
    _build_reg_map, _augment_with_local_vars, _augment_with_callee_regs,
    _augment_with_xram_params, _augment_with_callee_xram_params,
)
from pseudo8051.passes.typesimplify._simplify import _simplify, _simplify_once
from pseudo8051.passes.typesimplify._post     import (
    _consolidate_xram_local_loads, _collapse_dpl_dph,
    _fold_and_prune_setups, _fold_call_arg_pairs, _propagate_values,
    _simplify_carry_comparison, _simplify_cjne_jnc,
    _simplify_orl_zero_check,
    _prune_orphaned_dptr_inc,
    _fold_return_chains,
    _fold_xram_call_args,
)
from pseudo8051.passes.typesimplify._enum_resolve import _resolve_enum_consts
from pseudo8051.constants import dbg


class TypeAwareSimplifier(OptimizationPass):
    """
    Replace register-level expressions with typed variable names and collapse
    common multi-byte patterns via the registered Pattern list.
    """

    def run(self, func: Function) -> None:
        from pseudo8051.prototypes import get_proto
        live_in = getattr(func.entry_block, "live_in", frozenset())

        proto = get_proto(func.name)
        if proto is not None:
            reg_map = _build_reg_map(proto, live_in)
            dbg("typesimp", f"{func.name}: caller proto found, "
                            f"live_in={sorted(live_in)}  "
                            f"reg_map keys={list(reg_map.keys())}")
        else:
            reg_map = {}
            # Synthesize parameter entries from liveness when no prototype exists (issue 2.3)
            for idx, reg in enumerate(r for r in PARAM_REG_ORDER if r in live_in):
                info = VarInfo(f"arg{idx + 1}", "uint8_t", (reg,), is_param=True)
                reg_map[reg] = info
            dbg("typesimp", f"{func.name}: no caller proto, "
                            "scanning callee prototypes for register mappings")

        reg_map = _augment_with_local_vars(func.ea, reg_map)
        reg_map = _augment_with_xram_params(func.ea, reg_map)

        # Fall back to global callee-reg/XRAM scan when AnnotationPass hasn't run
        # (e.g. unit tests that call TypeAwareSimplifier directly).
        # When AnnotationPass has run, callee XRAM params are handled per-call-site
        # via _backward_annotate_xram_call → call_arg_ann on the XRAM write node.
        if not getattr(func, "_annotation_pass_ran", False):
            reg_map = _augment_with_callee_regs(func.hir, reg_map)
            reg_map = _augment_with_callee_xram_params(func.hir, reg_map)

        if not reg_map:
            dbg("typesimp", f"{func.name}: no register mappings found, running structural patterns only")

        dbg("typesimp", f"{func.name}: final reg_map={list(reg_map.keys())}")
        reg_map["__n__"] = [0]
        func.hir = _simplify(func.hir, reg_map)
        func.hir = _consolidate_xram_local_loads(func.hir, reg_map)
        func.hir = _simplify_once(func.hir, reg_map)
        func.hir = _fold_call_arg_pairs(func.hir, reg_map)
        func.hir = _collapse_dpl_dph(func.hir, reg_map)
        func.hir = _fold_and_prune_setups(func.hir, reg_map)
        # Remove cross-block nop-gotos in the assembled HIR (IfGoto whose target label
        # is the immediately following node in the flat list).  Must run before
        # _propagate_values so the resulting ExprStmt(cond) counts as a single use of
        # any register in the condition, enabling multi-use forward propagation below.
        from pseudo8051.passes.ifelse import _remove_nop_gotos
        func.hir = _remove_nop_gotos(func.hir)
        func.hir = _propagate_values(func.hir, reg_map)
        func.hir = _prune_orphaned_dptr_inc(func.hir)
        func.hir = _fold_xram_call_args(func.hir)
        func.hir = _fold_and_prune_setups(func.hir, reg_map)
        func.hir = _simplify_carry_comparison(func.hir)
        func.hir = _fold_and_prune_setups(func.hir, reg_map)  # clean up setups dead after SUBB16 collapse
        # Collapse CJNE(nop-goto) + JNC/JC → typed comparison (e.g. expr >= const).
        func.hir = _simplify_cjne_jnc(func.hir)
        func.hir = _simplify_orl_zero_check(func.hir)

        func.hir = collapse_mb_assigns(func.hir)

        # Prepend C-style declarations for any XRAM local variables.
        seen: set = set()
        local_decls = []
        for vinfo in reg_map.values():
            if (isinstance(vinfo, VarInfo)
                    and vinfo.xram_sym and not vinfo.is_byte_field
                    and not vinfo.is_param
                    and vinfo.name not in seen):
                seen.add(vinfo.name)
                local_decls.append(vinfo)
        if local_decls:
            local_decls.sort(key=lambda v: v.xram_sym)
            func.hir = [
                VarDecl(func.ea, v.type, v.name, v.xram_sym, v.xram_addr)
                for v in local_decls
            ] + func.hir

        # Fold Assign(ret_reg, expr); ReturnStmt(ret_reg) → ReturnStmt(expr) (issue 1)
        from pseudo8051.prototypes import expand_regs as _expand_regs
        ret_regs = (_expand_regs(tuple(proto.return_regs), proto.return_type)
                    if proto and proto.return_regs
                    else tuple(getattr(func, "return_registers", [])))
        if ret_regs:
            func.hir = _fold_return_chains(func.hir, ret_regs, reg_map)

        # Replace Const values with enum member names where the type context is an IDA enum.
        func.hir = _resolve_enum_consts(func.hir, reg_map)

        # Remove Label nodes in the assembled HIR that are no longer referenced
        # by any goto (can be orphaned by nop-goto removal + CJNE/JNC collapsing).
        from pseudo8051.passes.ifelse import _collect_goto_targets, _drop_dead_labels
        _live_labels = _collect_goto_targets(func.hir)
        func.hir = _drop_dead_labels(func.hir, _live_labels)

        from pseudo8051.passes.debug_dump import dump_pass_hir
        dump_pass_hir("typesimp", func.hir, func.name)
