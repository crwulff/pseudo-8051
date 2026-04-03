"""
locals_ui.py — IDA right-click menu for managing per-function XRAM locals
and register annotations.

Exports:
    _register_local_actions()
        Call once at module load to register the IDA actions.

    setup_popup(form, popup_handle, func_ea, func_name, viewer)
        Call from PseudocodeViewer.OnPopup() to attach the actions to the
        context menu.
"""

import sys

import ida_kernwin

# Context written by setup_popup() just before IDA invokes an action handler.
_popup_ctx: dict = {"func_ea": 0, "func_name": "", "viewer": None}


class _LocalManageAction(ida_kernwin.action_handler_t):
    """Open a table dialog to add/edit/delete XRAM local variables."""

    def activate(self, ctx) -> int:
        from pseudo8051.locals    import get_locals
        from pseudo8051.ui_dialogs import LocalsTableDialog
        from PyQt5.QtWidgets      import QApplication, QDialog
        func_ea = _popup_ctx["func_ea"]
        viewer  = _popup_ctx["viewer"]
        if not func_ea:
            return 1
        dlg = LocalsTableDialog(func_ea, get_locals(func_ea),
                                parent=QApplication.activeWindow())
        if dlg.exec_() == QDialog.Accepted and viewer is not None:
            viewer.Show(func_ea)
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


class _RegAnnManageAction(ida_kernwin.action_handler_t):
    """Open a table dialog to add/edit/delete register annotations."""

    def activate(self, ctx) -> int:
        from pseudo8051.reganns   import get_reganns
        from pseudo8051.ui_dialogs import RegAnnsTableDialog
        from PyQt5.QtWidgets      import QApplication, QDialog
        func_ea = _popup_ctx["func_ea"]
        viewer  = _popup_ctx["viewer"]
        if not func_ea:
            return 1
        dlg = RegAnnsTableDialog(func_ea, get_reganns(func_ea),
                                 parent=QApplication.activeWindow())
        if dlg.exec_() == QDialog.Accepted and viewer is not None:
            viewer.Show(func_ea)
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


class _XRAMParamManageAction(ida_kernwin.action_handler_t):
    """Open a table dialog to add/edit/delete XRAM parameters."""

    def activate(self, ctx) -> int:
        from pseudo8051.xram_params import get_xram_params
        from pseudo8051.ui_dialogs  import XRAMParamsTableDialog
        from PyQt5.QtWidgets        import QApplication, QDialog
        func_ea = _popup_ctx["func_ea"]
        viewer  = _popup_ctx["viewer"]
        if not func_ea:
            return 1
        dlg = XRAMParamsTableDialog(func_ea, get_xram_params(func_ea),
                                    parent=QApplication.activeWindow())
        if dlg.exec_() == QDialog.Accepted and viewer is not None:
            viewer.Show(func_ea)
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


class _ExprTreeAction(ida_kernwin.action_handler_t):
    """Toggle inline HIR node annotations on all pseudocode lines."""

    def activate(self, ctx) -> int:
        viewer  = _popup_ctx["viewer"]
        func_ea = _popup_ctx["func_ea"]
        if viewer is None or not func_ea:
            return 1
        viewer._annotate_nodes = not getattr(viewer, '_annotate_nodes', False)
        viewer.Show(func_ea)
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
    """Attach local-variable and register-annotation actions to the right-click menu."""
    _popup_ctx["func_ea"]   = func_ea
    _popup_ctx["func_name"] = func_name
    _popup_ctx["viewer"]    = viewer

    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:local_manage",
                                       "XRAM locals/")
    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:xram_param_manage",
                                       "XRAM parameters/")
    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:regann_manage",
                                       "Register annotations/")

    _c = sys.modules.get("pseudo8051.constants")
    label = "View constants as decimal" if (
        _c is None or getattr(_c, "USE_HEX", True)
    ) else "View constants as hexadecimal"
    ida_kernwin.update_action_label("pseudo8051:toggle_hex", label)
    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:toggle_hex", "")

    annotating = getattr(_popup_ctx["viewer"], '_annotate_nodes', False)
    ann_label = "Hide HIR node annotations" if annotating else "Annotate HIR nodes"
    ida_kernwin.update_action_label("pseudo8051:expr_tree", ann_label)
    ida_kernwin.attach_action_to_popup(form, popup_handle,
                                       "pseudo8051:expr_tree", "")


def _register_local_actions() -> None:
    """Register (or re-register after a reload) the UI actions."""
    _defs = [
        ("pseudo8051:local_manage",      "Manage\u2026",          _LocalManageAction()),
        ("pseudo8051:xram_param_manage", "Manage\u2026",          _XRAMParamManageAction()),
        ("pseudo8051:regann_manage",     "Manage\u2026",          _RegAnnManageAction()),
        ("pseudo8051:toggle_hex",        "View constants as decimal", _HexToggleAction()),
        ("pseudo8051:expr_tree",         "Annotate HIR nodes",    _ExprTreeAction()),
    ]
    for name, label, handler in _defs:
        ida_kernwin.unregister_action(name)
        ida_kernwin.register_action(ida_kernwin.action_desc_t(name, label, handler))
