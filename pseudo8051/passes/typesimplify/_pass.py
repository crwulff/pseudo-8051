"""
passes/typesimplify/_pass.py — TypeAwareSimplifier pass class.
"""

from pseudo8051.ir.hir    import Statement
from pseudo8051.passes    import OptimizationPass
from pseudo8051.passes.patterns.mb_assign import collapse_mb_assigns
from pseudo8051.passes.patterns._utils  import VarInfo
from pseudo8051.ir.function import Function

from pseudo8051.passes.typesimplify._regmap   import (
    _build_reg_map, _augment_with_local_vars, _augment_with_callee_regs,
)
from pseudo8051.passes.typesimplify._simplify import _simplify, _simplify_once
from pseudo8051.passes.typesimplify._post     import (
    _consolidate_xram_local_loads, _collapse_dpl_dph, _fold_and_prune_setups,
)
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
            dbg("typesimp", f"{func.name}: no caller proto, "
                            "scanning callee prototypes for register mappings")

        reg_map = _augment_with_local_vars(func.ea, reg_map)
        reg_map = _augment_with_callee_regs(func.hir, reg_map)

        if not reg_map:
            dbg("typesimp", f"{func.name}: no register mappings found, running structural patterns only")

        dbg("typesimp", f"{func.name}: final reg_map={list(reg_map.keys())}")
        reg_map["__n__"] = [0]
        func.hir = _simplify(func.hir, reg_map)
        func.hir = _consolidate_xram_local_loads(func.hir, reg_map)
        func.hir = _simplify_once(func.hir, reg_map)
        func.hir = _collapse_dpl_dph(func.hir, reg_map)
        func.hir = _fold_and_prune_setups(func.hir, reg_map)

        func.hir = collapse_mb_assigns(func.hir)

        # Prepend C-style declarations for any XRAM local variables.
        seen: set = set()
        local_decls = []
        for vinfo in reg_map.values():
            if (isinstance(vinfo, VarInfo)
                    and vinfo.xram_sym and not vinfo.is_byte_field
                    and vinfo.name not in seen):
                seen.add(vinfo.name)
                local_decls.append(vinfo)
        if local_decls:
            local_decls.sort(key=lambda v: v.xram_sym)
            func.hir = [
                Statement(func.ea,
                          f"{v.type} {v.name};"
                          + (f"  /* {v.xram_sym} @ {hex(v.xram_addr)} */"
                             if v.xram_addr else ""))
                for v in local_decls
            ] + func.hir
