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

from pseudo8051.ir.function import Function


def _show_as_tab(widget, title: str, dest: str = "IDA View-A") -> None:
    """
    Display widget as a tab next to dest.
    IDA 9+: display_widget accepts (widget, WOPN_DP_TAB, dest_title).
    IDA 7/8: use set_dock_pos then display_widget(widget, 0).
    """
    try:
        ida_kernwin.display_widget(widget, _WOPN_DP_TAB, dest)
    except TypeError:
        # Older IDA: display_widget takes only two arguments
        ida_kernwin.set_dock_pos(title, dest, _DP_INSIDE)
        ida_kernwin.display_widget(widget, 0)


class PseudocodeViewer(ida_kernwin.simplecustviewer_t):
    """
    Dockable IDA window showing 8051 pseudocode.
    Double-clicking a line jumps the disassembly view to that address.
    """

    def Create(self, title: str = "8051 Pseudocode") -> bool:
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
            _show_as_tab(self.GetWidget(), "8051 Pseudocode")
            return

        func_name  = func.name
        params     = func.parameters
        params_str = ", ".join(params) if params else "—"

        self.AddLine(f"// Function: {func_name}  @  {hex(func_ea)}")
        self.AddLine(f"// Params:   {params_str}")
        self.AddLine("//")
        self.AddLine("")

        HEADER_LINES = 4
        lines = func.render()
        for i, (ea, text) in enumerate(lines):
            self._ea_map[HEADER_LINES + i] = ea
            self.AddLine(text)

        self.Refresh()
        _show_as_tab(self.GetWidget(), "8051 Pseudocode")

    def OnDblClick(self, _shift: int) -> bool:
        """Jump IDA disassembly view to the address of the clicked line."""
        ea = self._ea_map.get(self.GetLineNo())
        if ea is not None:
            idc.jumpto(ea)
        return True

    def OnClose(self) -> None:
        pass


def run_pseudocode_view() -> None:
    ea   = idc.here()
    func = ida_funcs.get_func(ea)
    if not func:
        ida_kernwin.warning("Cursor is not inside a function.\n"
                            "Place the cursor on a function and re-run.")
        return

    # Keep a module-level reference so Python's GC doesn't collect the viewer
    # while it is still displayed by IDA.
    global _8051_viewer
    _8051_viewer = PseudocodeViewer()

    if not _8051_viewer.Create("8051 Pseudocode"):
        ida_kernwin.warning("Failed to create pseudocode viewer.")
        return

    _8051_viewer.Show(func.start_ea)


run_pseudocode_view()
