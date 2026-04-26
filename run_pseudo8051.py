"""
run_pseudo8051.py — IDA Pro entry point for the OO 8051 pseudocode generator.

Usage: File → Script File → run_pseudo8051.py
       Place cursor inside a function first.

Re-imports all submodules in dependency order on every run so source edits
are picked up without restarting IDA.
"""

import os
import sys
import importlib

# Ensure IDAScripts/ is on sys.path so `pseudo8051` can be found as a package.
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# Reload order: leaves → roots (each module reloaded after its dependencies).
_RELOAD_ORDER = [
    "pseudo8051.constants",
    "pseudo8051.colorize",
    "pseudo8051.locals",
    "pseudo8051.iram_locals",
    "pseudo8051.xram_params",
    "pseudo8051.trampolines",
    "pseudo8051.prototypes",
    "pseudo8051.ir.expr._prec",
    "pseudo8051.ir.expr._base",
    "pseudo8051.ir.expr.regs",
    "pseudo8051.ir.expr.const",
    "pseudo8051.ir.expr.name",
    "pseudo8051.ir.expr.xram_ref",
    "pseudo8051.ir.expr.iram_ref",
    "pseudo8051.ir.expr.crom_ref",
    "pseudo8051.ir.expr.bin_op",
    "pseudo8051.ir.expr.unary_op",
    "pseudo8051.ir.expr.call",
    "pseudo8051.ir.expr.rot9op",
    "pseudo8051.ir.expr.rot8op",
    "pseudo8051.ir.expr.array_ref",
    "pseudo8051.ir.expr.paren",
    "pseudo8051.ir.expr.cast",
    "pseudo8051.ir.expr",
    "pseudo8051.ir.hir._base",
    "pseudo8051.ir.hir.assign",
    "pseudo8051.ir.hir.compound_assign",
    "pseudo8051.ir.hir.expr_stmt",
    "pseudo8051.ir.hir.return_stmt",
    "pseudo8051.ir.hir.if_goto",
    "pseudo8051.ir.hir.statement",
    "pseudo8051.ir.hir.goto_statement",
    "pseudo8051.ir.hir.break_stmt",
    "pseudo8051.ir.hir.continue_stmt",
    "pseudo8051.ir.hir.var_decl",
    "pseudo8051.ir.hir.computed_jump",
    "pseudo8051.ir.hir.label",
    "pseudo8051.ir.hir.if_node",
    "pseudo8051.ir.hir.while_node",
    "pseudo8051.ir.hir.for_node",
    "pseudo8051.ir.hir.do_while_node",
    "pseudo8051.ir.hir.switch_node",
    "pseudo8051.ir.hir",
    "pseudo8051.ir.operand",
    "pseudo8051.ir.instruction",
    "pseudo8051.handlers.mov",
    "pseudo8051.handlers.arithmetic",
    "pseudo8051.handlers.logic",
    "pseudo8051.handlers.branch",
    "pseudo8051.handlers.call",
    "pseudo8051.handlers",
    "pseudo8051.passes",           # defines OptimizationPass ABC
    "pseudo8051.passes.debug_dump",
    "pseudo8051.passes.chunk_inline",
    "pseudo8051.passes.simple_inline",
    "pseudo8051.passes.annotate",
    "pseudo8051.passes.rmw",
    "pseudo8051.passes.loops",
    "pseudo8051.passes.jmptable",
    "pseudo8051.passes.switch",
    "pseudo8051.passes.ifelse",
    "pseudo8051.passes.cjne_switch",
    "pseudo8051.passes.patterns.base",
    "pseudo8051.passes.patterns._utils",
    "pseudo8051.passes.patterns.sign_bit",
    "pseudo8051.passes.patterns.neg16",
    "pseudo8051.passes.patterns.const_group",
    "pseudo8051.passes.patterns.xram_group_read",
    "pseudo8051.passes.patterns.xram_local_write",
    "pseudo8051.passes.patterns.mb_add",
    "pseudo8051.passes.patterns.mb_incdec",
    "pseudo8051.passes.patterns.retval",
    "pseudo8051.passes.patterns.reg_copy_group",
    "pseudo8051.passes.patterns.xch_copy",
    "pseudo8051.passes.patterns.reg_inc",
    "pseudo8051.passes.patterns.accum_relay",
    "pseudo8051.passes.patterns.rol_switch",
    "pseudo8051.passes.patterns.accum_fold",
    "pseudo8051.passes.patterns.mb_assign",
    "pseudo8051.passes.patterns",
    "pseudo8051.passes.typesimplify._regmap",
    "pseudo8051.passes.typesimplify._simplify",
    "pseudo8051.passes.typesimplify._dptr",
    "pseudo8051.passes.typesimplify._xram_loads",
    "pseudo8051.passes.typesimplify._setup_fold",
    "pseudo8051.passes.typesimplify._return_fold",
    "pseudo8051.passes.typesimplify._propagate",
    "pseudo8051.passes.typesimplify._carry",
    "pseudo8051.passes.typesimplify._xram_call_args",
    "pseudo8051.passes.typesimplify._post",
    "pseudo8051.enum_resolve",
    "pseudo8051.passes.typesimplify._enum_resolve",
    "pseudo8051.passes.typesimplify._pass",
    "pseudo8051.passes.typesimplify",
    "pseudo8051.passes.switchcomment",
    "pseudo8051.analysis.constprop",
    "pseudo8051.analysis.liveness",
    "pseudo8051.ir.cpstate",
    "pseudo8051.ir.basicblock",
    "pseudo8051.ir.function",
    "pseudo8051.ui_dialogs",
    "pseudo8051.locals_ui",
    "pseudo8051.detail_viewer",
    "pseudo8051",
]

# If a previous session loaded pseudo8051.ir.hir as a flat module (before the
# package refactor), it will be cached in sys.modules without a __path__ and
# Python will refuse to import submodules from it.  Purge all stale entries.
_hir = sys.modules.get("pseudo8051.ir.hir")
if _hir is not None and not hasattr(_hir, "__path__"):
    for _k in [k for k in list(sys.modules) if k == "pseudo8051.ir.hir"
               or k.startswith("pseudo8051.ir.hir.")]:
        del sys.modules[_k]
del _hir

# First import (no-op if already loaded); then reload to pick up edits.
for _mod in _RELOAD_ORDER:
    if _mod not in sys.modules:
        importlib.import_module(_mod)
    else:
        importlib.reload(sys.modules[_mod])

sys.modules["pseudo8051"].run_pseudocode_view()
