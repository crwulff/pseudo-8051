"""
pseudo8051/__init__.py — Entry point for the OO 8051 pseudocode generator.

Run via File → Script File in IDA Pro.  Place cursor inside a function first.

Architecture overview:
    pseudo8051/
    ├── constants.py          SFR names, register IDs, address helpers
    ├── ir/
    │   ├── operand.py        Operand — wraps one IDA operand, renders itself
    │   ├── instruction.py    Instruction — wraps insn_t; MnemonicHandler ABC
    │   ├── basicblock.py     BasicBlock — sequence of Instructions + slots
    │   ├── function.py       Function — block graph + pass runner
    │   └── hir.py            HIR nodes: Statement, Label, IfNode, WhileNode…
    ├── handlers/             Per-mnemonic handler implementations
    ├── analysis/
    │   ├── constprop.py      ConstantPropagation (forward dataflow)
    │   └── liveness.py       LivenessAnalysis (backward dataflow)
    └── passes/
        ├── rmw.py            RMWCollapser
        ├── loops.py          LoopStructurer
        └── ifelse.py         IfElseStructurer
"""

import os as _os, sys as _sys
_scripts_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _scripts_dir not in _sys.path:
    _sys.path.insert(0, _scripts_dir)
del _os, _sys, _scripts_dir

import string
import sys

import ida_funcs
import ida_kernwin
import idc

# IDA 9 SDK: WOPN_DP_TAB = (DP_TAB << WOPN_DP_SHIFT) where WOPN_DP_SHIFT = 16.
# DP_TAB itself is typically 6 in the kernwin.hpp enum.
# Fall back to computed values if the Python bindings don't expose them directly.
_DP_TAB      = getattr(ida_kernwin, 'DP_TAB',      6)
_WOPN_DP_TAB = getattr(ida_kernwin, 'WOPN_DP_TAB', _DP_TAB << 16)

# IDA 7/8 fallback constants
_DP_INSIDE = getattr(ida_kernwin, 'DP_INSIDE', 5)

_BASE_TITLE = "8051 Pseudocode"

from pseudo8051.ir.function import Function

# Viewer registry that survives module reloads: keyed by tab title.
# We store it on sys so it is not reset when this module is reloaded.
if not hasattr(sys, '_pseudo8051_viewers'):
    sys._pseudo8051_viewers: dict = {}


def _show_as_tab(widget, title: str) -> None:
    """
    Display widget as a tab.  Prefer to dock next to an existing pseudocode
    tab; fall back to IDA View-A if none is open yet.
    IDA 9+: display_widget accepts (widget, WOPN_DP_TAB, dest_title).
    IDA 7/8: use set_dock_pos then display_widget(widget, 0).
    """
    # Find an already-open pseudocode tab to dock next to
    dest = "IDA View-A"
    for existing_title in sys._pseudo8051_viewers:
        if ida_kernwin.find_widget(existing_title) is not None:
            dest = existing_title
            break

    try:
        ida_kernwin.display_widget(widget, _WOPN_DP_TAB, dest)
    except TypeError:
        # Older IDA: display_widget takes only two arguments
        ida_kernwin.set_dock_pos(title, dest, _DP_INSIDE)
        ida_kernwin.display_widget(widget, 0)


def _next_title() -> str:
    """
    Return the next available tab title of the form "8051 Pseudocode-A",
    "8051 Pseudocode-B", … by checking which widgets are currently open.
    """
    for letter in string.ascii_uppercase:
        title = f"{_BASE_TITLE}-{letter}"
        if ida_kernwin.find_widget(title) is None:
            return title
    return f"{_BASE_TITLE}-?"   # all 26 slots occupied (shouldn't happen)


class PseudocodeViewer(ida_kernwin.simplecustviewer_t):
    """
    Dockable IDA window showing 8051 pseudocode.
    Double-clicking a line jumps the disassembly view to that address.
    """

    def Create(self, title: str) -> bool:
        self._title = title
        return super().Create(title)

    def Show(self, func_ea: int) -> None:
        """(Re-)generate pseudocode for func_ea and display the window."""
        self.ClearLines()
        self._ea_map = {}   # viewer line number → instruction EA

        try:
            func = Function(func_ea)
        except Exception as e:
            self.AddLine(f"/* ERROR building IR: {e} */")
            self.Refresh()
            _show_as_tab(self.GetWidget(), self._title)
            return

        lines = func.render()
        for i, (ea, text) in enumerate(lines):
            self._ea_map[i] = ea
            self.AddLine(text)

        self.Refresh()
        _show_as_tab(self.GetWidget(), self._title)

    def OnDblClick(self, _shift: int) -> bool:
        """Jump IDA disassembly view to the address of the clicked line."""
        ea = self._ea_map.get(self.GetLineNo())
        if ea is not None:
            idc.jumpto(ea)
        return True

    def OnClose(self) -> None:
        sys._pseudo8051_viewers.pop(getattr(self, '_title', None), None)


def run_pseudocode_view() -> None:
    ea   = idc.here()
    func = ida_funcs.get_func(ea)
    if not func:
        ida_kernwin.warning("Cursor is not inside a function.\n"
                            "Place the cursor on a function and re-run.")
        return

    title = _next_title()
    viewer = PseudocodeViewer()

    if not viewer.Create(title):
        ida_kernwin.warning("Failed to create pseudocode viewer.")
        return

    # Store in the persistent registry so GC doesn't collect it
    sys._pseudo8051_viewers[title] = viewer
    viewer.Show(func.start_ea)
