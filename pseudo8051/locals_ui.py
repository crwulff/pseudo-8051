"""
locals_ui.py — IDA right-click menu for managing per-function XRAM locals.

Exports:
    _register_local_actions()
        Call once at module load to register the three IDA actions.

    setup_popup(form, popup_handle, func_ea, func_name, viewer)
        Call from PseudocodeViewer.OnPopup() to attach the actions to the
        context menu under an "XRAM locals" submenu.
"""

import sys

import ida_kernwin

# Context written by setup_popup() just before IDA invokes an action handler.
_popup_ctx: dict = {"func_ea": 0, "func_name": "", "viewer": None}


class _LocalsChooser(ida_kernwin.Choose):
    """Modal chooser that lists the XRAM locals for one function."""

    def __init__(self, title: str, locals_list):
        super().__init__(title,
                         [["Address", 10], ["Type", 12], ["Name", 24]],
                         flags=ida_kernwin.Choose.CH_MODAL)
        self._items = locals_list

    def OnGetSize(self) -> int:
        return len(self._items)

    def OnGetLine(self, n: int):
        lv = self._items[n]
        return [f"{lv.addr:#06x}", lv.type, lv.name]

    def OnGetIcon(self, n: int) -> int:
        return -1


def _chooser_pick(title: str, locs) -> int:
    """Show a modal chooser; return selected index or -1 if cancelled."""
    result = _LocalsChooser(title, locs).Show(True)
    if result is None:
        return -1
    if isinstance(result, (list, tuple)):
        return result[0] if result else -1
    return int(result)


class _LocalAddAction(ida_kernwin.action_handler_t):
    """Add or edit an XRAM local variable for the current function."""

    def activate(self, ctx) -> int:
        from pseudo8051.locals import get_locals, set_local
        func_ea = _popup_ctx["func_ea"]
        if not func_ea:
            return 1
        addr_s = ida_kernwin.ask_str("", 0, "XRAM address (e.g. 0xdc8a):")
        if not addr_s:
            return 1
        try:
            addr = int(addr_s, 0)
        except ValueError:
            ida_kernwin.warning(f"Invalid address: {addr_s!r}")
            return 1
        existing = {lv.addr: lv for lv in get_locals(func_ea)}
        lv = existing.get(addr)
        name = ida_kernwin.ask_str(lv.name if lv else "", 0, "Variable name:")
        if not name:
            return 1
        type_s = ida_kernwin.ask_str(lv.type if lv else "uint8_t", 0,
                                     "C type (e.g. int16_t, uint32_t):")
        if not type_s:
            return 1
        set_local(func_ea, addr, name.strip(), type_s.strip())
        viewer = _popup_ctx["viewer"]
        if viewer is not None:
            viewer.Show(func_ea)
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


class _LocalDelAction(ida_kernwin.action_handler_t):
    """Remove an XRAM local variable from the current function."""

    def activate(self, ctx) -> int:
        from pseudo8051.locals import get_locals, del_local
        func_ea   = _popup_ctx["func_ea"]
        func_name = _popup_ctx["func_name"]
        if not func_ea:
            return 1
        locs = get_locals(func_ea)
        if not locs:
            ida_kernwin.info(f"No locals declared in {func_name}.")
            return 1
        n = _chooser_pick(f"Remove local — {func_name}", locs)
        if n < 0:
            return 1
        lv = locs[n]
        del_local(func_ea, lv.addr)
        print(f"pseudo8051: removed {lv.type} {lv.name} @ {lv.addr:#06x}")
        viewer = _popup_ctx["viewer"]
        if viewer is not None:
            viewer.Show(func_ea)
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


class _LocalListAction(ida_kernwin.action_handler_t):
    """Show the XRAM local variables declared for the current function."""

    def activate(self, ctx) -> int:
        from pseudo8051.locals import get_locals
        func_ea   = _popup_ctx["func_ea"]
        func_name = _popup_ctx["func_name"]
        if not func_ea:
            return 1
        locs = get_locals(func_ea)
        if not locs:
            ida_kernwin.info(f"No locals declared in {func_name}.")
            return 1
        _chooser_pick(f"Locals — {func_name}", locs)
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


class _HexToggleAction(ida_kernwin.action_handler_t):
    """Toggle integer constants between hexadecimal and decimal display."""

    def activate(self, ctx) -> int:
        _c = sys.modules.get("pseudo8051.constants")
        if _c is None:
            return 1
        _c.USE_HEX = not _c.USE_HEX
        viewer = _popup_ctx["viewer"]
        func_ea = _popup_ctx["func_ea"]
        if viewer is not None and func_ea:
            viewer.Show(func_ea)
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


def setup_popup(form, popup_handle,
                func_ea: int, func_name: str, viewer) -> None:
    """Attach XRAM-local actions to the right-click context menu."""
    _popup_ctx["func_ea"]   = func_ea
    _popup_ctx["func_name"] = func_name
    _popup_ctx["viewer"]    = viewer
    p = "XRAM locals/"
    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:local_add",  p)
    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:local_del",  p)
    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:local_list", p)
    _c = sys.modules.get("pseudo8051.constants")
    label = "View constants as decimal" if (
        _c is None or getattr(_c, "USE_HEX", True)
    ) else "View constants as hexadecimal"
    ida_kernwin.update_action_label("pseudo8051:toggle_hex", label)
    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:toggle_hex", "")


def _register_local_actions() -> None:
    """Register (or re-register after a reload) the local-variable UI actions."""
    _defs = [
        ("pseudo8051:local_add",  "Add / edit local variable\u2026", _LocalAddAction()),
        ("pseudo8051:local_del",  "Remove local variable\u2026",     _LocalDelAction()),
        ("pseudo8051:local_list", "List local variables",            _LocalListAction()),
        ("pseudo8051:toggle_hex", "View constants as decimal",       _HexToggleAction()),
    ]
    for name, label, handler in _defs:
        ida_kernwin.unregister_action(name)
        ida_kernwin.register_action(ida_kernwin.action_desc_t(name, label, handler))
