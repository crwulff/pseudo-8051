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

XRAM local variable declarations (stored in IDA netnode, persistent per function):

    import pseudo8051
    pseudo8051.set_local(here(), 0xdc8a, 'count', 'int16_t')
    pseudo8051.list_locals(here())
    pseudo8051.del_local(here(), 0xdc8a)

XRAM parameters (extra parameters passed via fixed XRAM addresses):

    pseudo8051.set_xram_param(here(), 0xdc8a, 'extra', 'uint8_t')
    pseudo8051.list_xram_params(here())
    pseudo8051.del_xram_param(here(), 0xdc8a)

Register annotations (per-instruction, stored in IDA netnode):

    pseudo8051.set_regann(here(), here(), 'R6,R7', 'count', 'uint16_t')
    pseudo8051.list_reganns(here())
    pseudo8051.del_regann(here(), here(), 'R6,R7')

'here()' may be any address inside the target function.
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

from pseudo8051.ir.expr    import Expr
from pseudo8051.ir.function import Function
from pseudo8051.locals      import set_local, del_local, list_locals          # noqa: F401
from pseudo8051.xram_params import set_xram_param, del_xram_param, list_xram_params  # noqa: F401
from pseudo8051.reganns     import set_regann, del_regann, list_reganns       # noqa: F401
from pseudo8051.locals_ui import setup_popup, _register_local_actions

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


def _walk_expr(expr, lines: list, indent: str) -> None:
    """Append indented tree lines for one Expr node, recursing into children."""
    name = type(expr).__name__
    if name == 'BinOp':
        lines.append(f"{indent}BinOp {expr.op!r}")
        _walk_expr(expr.lhs, lines, indent + "  ")
        _walk_expr(expr.rhs, lines, indent + "  ")
    elif name == 'UnaryOp':
        suffix = " (post)" if expr.post else ""
        lines.append(f"{indent}UnaryOp {expr.op!r}{suffix}")
        _walk_expr(expr.operand, lines, indent + "  ")
    elif name in ('XRAMRef', 'IRAMRef', 'CROMRef'):
        lines.append(f"{indent}{name}")
        _walk_expr(expr.inner, lines, indent + "  ")
    elif name == 'Cast':
        lines.append(f"{indent}Cast {expr.type_str!r}")
        _walk_expr(expr.inner, lines, indent + "  ")
    elif name == 'Call':
        lines.append(f"{indent}Call {expr.func_name!r}")
        for arg in expr.args:
            _walk_expr(arg, lines, indent + "  ")
    else:  # Reg, Const, Name, RegGroup — already have useful reprs
        lines.append(f"{indent}{repr(expr)}")


def _hir_node_tree(node) -> list:
    """
    Return annotation lines (plain strings, no '//' prefix) for one HIR node.
    First line is the class name; subsequent lines show field names and their
    expression trees, indented two spaces per level.
    """
    cls  = type(node).__name__
    out  = [cls]

    def expr_field(label, val):
        """Append 'label:' plus a recursive expr tree, or a plain repr."""
        if isinstance(val, Expr):
            sub = []
            _walk_expr(val, sub, "")
            out.append(f"  {label}:")
            for ln in sub:
                out.append(f"    {ln}")
        elif val is not None:
            out.append(f"  {label}: {val!r}")

    if cls == 'Assign':
        expr_field("lhs", node.lhs)
        expr_field("rhs", node.rhs)
    elif cls == 'CompoundAssign':
        expr_field("lhs", node.lhs)
        out.append(f"  op: {node.op!r}")
        expr_field("rhs", node.rhs)
    elif cls == 'ExprStmt':
        expr_field("expr", node.expr)
    elif cls == 'ReturnStmt':
        expr_field("value", node.value)
    elif cls == 'IfGoto':
        expr_field("cond", node.cond)
        out.append(f"  label: {node.label!r}")
    elif cls == 'Statement':
        out.append(f"  text: {node.text!r}")
    elif cls == 'GotoStatement':
        out.append(f"  label: {node.label!r}")
    elif cls == 'Label':
        out.append(f"  name: {node.name!r}")
    elif cls == 'IfNode':
        expr_field("cond", node.condition)
        out.append(f"  then: {len(node.then_nodes)} nodes")
        out.append(f"  else: {len(node.else_nodes)} nodes")
    elif cls == 'WhileNode':
        expr_field("cond", node.condition)
        out.append(f"  body: {len(node.body_nodes)} nodes")
    elif cls == 'ForNode':
        expr_field("init", node.init)
        expr_field("cond", node.condition)
        expr_field("update", node.update)
        out.append(f"  body: {len(node.body_nodes)} nodes")
    elif cls == 'SwitchNode':
        expr_field("subject", node.subject)
        out.append(f"  cases: {len(node.cases)}")
        if node.default_label is not None:
            out.append(f"  default: {node.default_label!r}")
    return out


def _collect_line_chains(nodes, node_map: dict, offset: int, ancestors: tuple) -> int:
    """
    Recursively map every viewer line number to its full HIR containment chain.

    Each entry is a tuple of HIRNodes from outermost to innermost.
    Container lines (opening/closing braces, 'else') stay at the outer chain;
    interior lines are overwritten with a deeper chain by the recursive call.
    Returns the next offset after all nodes.
    """
    for node in nodes:
        node_lines = node.render(indent=0)
        chain = ancestors + (node,)
        for k in range(len(node_lines)):
            node_map[offset + k] = chain
        # Recurse into child lists (overwrites interior lines with deeper chain)
        name = type(node).__name__
        sub = offset + 1   # first line after opening line
        if name == 'IfNode':
            sub = _collect_line_chains(node.then_nodes, node_map, sub, chain)
            if node.else_nodes:
                sub += 1   # "} else {"
                sub = _collect_line_chains(node.else_nodes, node_map, sub, chain)
        elif name in ('WhileNode', 'ForNode'):
            _collect_line_chains(node.body_nodes, node_map, sub, chain)
        # SwitchNode has no child HIR nodes to recurse into
        offset += len(node_lines)
    return offset


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
        self._ea_map   = {}   # viewer line number → instruction EA
        self._func_ea  = func_ea
        self._func_name = hex(func_ea)   # overwritten below on success

        try:
            func = Function(func_ea)
        except Exception as e:
            self.AddLine(f"/* ERROR building IR: {e} */")
            self.Refresh()
            _show_as_tab(self.GetWidget(), self._title)
            return

        self._func_name = func.name
        lines = func.render()

        # Build viewer-line → node chain map (lines 0="sig", 1="{" are skipped)
        self._node_map: dict = {}
        _collect_line_chains(func.hir, self._node_map, offset=2, ancestors=())

        annotate = getattr(self, '_annotate_nodes', False)
        vline = 0   # actual viewer line counter (differs from i when annotating)
        for i, (ea, text) in enumerate(lines):
            self._ea_map[vline] = ea
            self.AddLine(text)
            vline += 1
            if annotate:
                chain = self._node_map.get(i)
                if chain:
                    for ann in _hir_node_tree(chain[-1]):
                        self._ea_map[vline] = ea
                        self.AddLine(f"    // {ann}")
                        vline += 1

        self.Refresh()
        _show_as_tab(self.GetWidget(), self._title)

    def OnPopup(self, form, popup_handle) -> bool:
        """Add XRAM-local management items to the right-click context menu."""
        setup_popup(form, popup_handle,
                    getattr(self, "_func_ea",   0),
                    getattr(self, "_func_name", ""),
                    self)
        return False

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


_register_local_actions()
