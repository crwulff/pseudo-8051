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

def _get_insn_comment(ea: int) -> str:
    """Return the user-visible comment for instruction ea, or '' if none.

    Tries in order:
      1. idc.get_cmt(ea, 0) — regular comment directly at ea
      2. idc.get_cmt(ea, 1) — repeatable comment directly at ea
      3. If ea has a code reference to a function start, the callee's
         repeatable comment — this surfaces comments written on function
         definitions that IDA shows at call sites in the disassembly view.
    """
    cmt0 = idc.get_cmt(ea, 0)
    cmt1 = idc.get_cmt(ea, 1)
    print(f"[cmt] ea={hex(ea)} cmt0={cmt0!r} cmt1={cmt1!r}")
    cmt = cmt0 or cmt1
    if cmt:
        print(f"[cmt] -> direct comment at {hex(ea)}: {cmt!r}")
        return cmt
    try:
        import ida_xref, ida_funcs, idautils
        for callee_ea in idautils.CodeRefsFrom(ea, flow=False):
            print(f"[cmt] ea={hex(ea)} callee_ea={hex(callee_ea)}")
            fn = ida_funcs.get_func(callee_ea)
            print(f"[cmt] fn={fn} fn.start_ea={hex(fn.start_ea) if fn else 'None'}")
            if fn is not None and fn.start_ea == callee_ea:
                cmt = idc.get_func_cmt(callee_ea, 1) or idc.get_cmt(callee_ea, 1)
                print(f"[cmt] callee repeatable comment: {cmt!r}")
                if cmt:
                    return cmt
    except Exception as e:
        print(f"[cmt] exception: {e}")
    return ''

from pseudo8051.ir.function import Function
from pseudo8051.colorize    import colorize
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



class _SwitchCaseView:
    """
    Sentinel stored as the innermost chain entry for case/default header lines
    within a SwitchNode.  Carries the parent SwitchNode and the case index
    (-1 = default) so the detail viewer can:
      • show the switch dispatch src_eas as "Instructions"
      • show just this case's body as "Children"
    """
    __slots__ = ('switch_node', 'case_index', 'ea', 'src_eas')

    def __init__(self, switch_node, case_index: int):
        self.switch_node = switch_node
        self.case_index  = case_index   # -1 means default
        self.ea          = switch_node.ea
        # Use per-case EAs if tracked; fall back to the whole switch's src_eas.
        if case_index == -1:
            eas = getattr(switch_node, 'default_src_eas', None)
        else:
            _cse = getattr(switch_node, 'case_src_eas', None)
            eas  = _cse[case_index] if (_cse and 0 <= case_index < len(_cse)) else None
        self.src_eas = eas if eas else switch_node.src_eas

    def case_label(self) -> str:
        if self.case_index == -1:
            return "default:"
        values, _ = self.switch_node.cases[self.case_index]
        return " ".join(f"case {v}:" for v in values)

    def case_body(self):
        """Body HIRNodes for this case arm, or [] for goto-form cases."""
        if self.case_index == -1:
            b = self.switch_node.default_body
        else:
            _, b = self.switch_node.cases[self.case_index]
        return b if isinstance(b, list) else []


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
        for _extra, _child_nodes in node.child_body_groups():
            sub += _extra
            sub = _collect_line_chains(_child_nodes, node_map, sub, chain)
        if name == 'SwitchNode':
            # sub = first line after "switch (...) {"
            for ci, (_, body) in enumerate(node.cases):
                case_chain = ancestors + (node, _SwitchCaseView(node, ci))
                if isinstance(body, str):
                    node_map[sub] = case_chain   # "case N: goto label;"
                    sub += 1
                else:
                    node_map[sub] = case_chain   # "case N:" header
                    sub += 1
                    sub = _collect_line_chains(body, node_map, sub, ancestors + (node,))
            if node.default_body is not None:
                node_map[sub] = ancestors + (node, _SwitchCaseView(node, -1))
                sub += 1
                _collect_line_chains(node.default_body, node_map, sub, ancestors + (node,))
            elif node.default_label is not None:
                node_map[sub] = ancestors + (node, _SwitchCaseView(node, -1))
            # closing "}" stays as chain (already set by initial loop above)
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
        _cmt_shown: set = set()   # EAs whose IDA comment has already been emitted
        for i, (ea, text) in enumerate(lines):
            self._ea_map[vline] = ea
            # Append IDA comment (regular or repeatable) to the first line for
            # this EA.  Multi-line comments produce extra indented viewer lines.
            # Also check src_eas of the HIR node so that folded instructions
            # (e.g. lcall absorbed into an assignment) surface their comments.
            cmt_lines = []
            chain = self._node_map.get(i)
            _eas_to_check = [ea]
            if chain:
                _node = chain[-1]
                for _src_ea in sorted(getattr(_node, 'src_eas', frozenset())):
                    if _src_ea not in _eas_to_check:
                        _eas_to_check.append(_src_ea)
            for _check_ea in _eas_to_check:
                if _check_ea not in _cmt_shown:
                    raw = _get_insn_comment(_check_ea)
                    if raw:
                        _cmt_shown.add(_check_ea)
                        cmt_lines = raw.splitlines()
                        break
            if cmt_lines:
                self.AddLine(colorize(f"{text}  // {cmt_lines[0]}"))
                vline += 1
                for extra in cmt_lines[1:]:
                    self._ea_map[vline] = ea
                    self.AddLine(colorize(f"  // {extra}"))
                    vline += 1
            else:
                self.AddLine(colorize(text))
                vline += 1
            if annotate:
                chain = self._node_map.get(i)
                if chain:
                    node = chain[-1]
                    anns = node.ann_lines() + node.node_ann_lines()
                    if anns:
                        anns[0] = f"{anns[0]} [{hex(node.ea)}]"
                    for ann in anns:
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

    def OnClick(self, _shift: int) -> bool:
        """Update the detail viewer (if open) when the cursor moves via mouse."""
        _maybe_update_detail(self)
        return False  # don't consume — let IDA handle selection

    def OnKeydown(self, key: int, shift: int) -> bool:
        """Update the detail viewer after arrow-key cursor movement."""
        # The cursor hasn't moved yet at this point; defer to the next
        # event-loop iteration so GetLineNo() reflects the new position.
        try:
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(0, lambda: _maybe_update_detail(self))
        except Exception:
            pass
        return False  # don't consume — let IDA handle the key

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


def _maybe_update_detail(viewer) -> None:
    """Update the paired detail viewer (if open) from the given pseudocode viewer."""
    dv_mod = sys.modules.get('pseudo8051.detail_viewer')
    if dv_mod is None:
        return
    pv_title     = getattr(viewer, '_title', '')
    suffix       = pv_title[len("8051 Pseudocode"):]
    detail_title = f"8051 Detail{suffix}"
    dv_viewers   = getattr(dv_mod, '_viewers', {})
    dv = dv_viewers.get(detail_title)
    if dv is not None and getattr(dv, '_visible', False):
        dv.update_from(viewer)


_register_local_actions()
