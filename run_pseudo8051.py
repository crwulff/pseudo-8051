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
    "pseudo8051.locals",
    "pseudo8051.prototypes",
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
    "pseudo8051.passes.rmw",
    "pseudo8051.passes.loops",
    "pseudo8051.passes.ifelse",
    "pseudo8051.passes.patterns.base",
    "pseudo8051.passes.patterns._utils",
    "pseudo8051.passes.patterns.sign_bit",
    "pseudo8051.passes.patterns.neg16",
    "pseudo8051.passes.patterns.const_group",
    "pseudo8051.passes.patterns.xram_group_read",
    "pseudo8051.passes.patterns.xram_local_write",
    "pseudo8051.passes.patterns.mb_add",
    "pseudo8051.passes.patterns.retval",
    "pseudo8051.passes.patterns.reg_copy_group",
    "pseudo8051.passes.patterns",
    "pseudo8051.passes.typesimplify",
    "pseudo8051.analysis.constprop",
    "pseudo8051.analysis.liveness",
    "pseudo8051.ir.basicblock",
    "pseudo8051.ir.function",
    "pseudo8051.locals_ui",
    "pseudo8051",
]

# First import (no-op if already loaded); then reload to pick up edits.
for _mod in _RELOAD_ORDER:
    if _mod not in sys.modules:
        importlib.import_module(_mod)
    else:
        importlib.reload(sys.modules[_mod])

sys.modules["pseudo8051"].run_pseudocode_view()
