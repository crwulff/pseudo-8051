"""
ui_dialogs.py — PyQt5 table dialogs for managing XRAM locals and register annotations.

LocalsTableDialog: editable table of (Address, Type, Name) for XRAM local variables.
RegAnnsTableDialog: editable table of (Address, Register(s), Name, Type) for reg annotations.
"""

from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore    import Qt
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)


# ── Shared base ────────────────────────────────────────────────────────────────

class _TableDialog(QDialog):
    """
    Base class for editable table dialogs.

    Subclasses must implement:
        _get_column_headers() -> List[str]
        _populate(data)       -- fill self._table from data
        _validate_row(row_data: List[str]) -> Optional[str]
            Return an error string, or None if valid.
        _sync_changes(original, current) -> Optional[str]
            Apply diffs; return an error string on failure, or None on success.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self.setMinimumHeight(350)

        headers = self._get_column_headers()
        self._table = QTableWidget(0, len(headers), self)
        self._table.setHorizontalHeaderLabels(headers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.DoubleClicked |
                                    QTableWidget.SelectedClicked |
                                    QTableWidget.EditKeyPressed)

        btn_add = QPushButton("Add Row", self)
        btn_add.clicked.connect(self._add_row)
        btn_del = QPushButton("Delete Row", self)
        btn_del.clicked.connect(self._delete_rows)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)

        row_btns = QHBoxLayout()
        row_btns.addWidget(btn_add)
        row_btns.addWidget(btn_del)
        row_btns.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(row_btns)
        layout.addWidget(btn_box)

    # -- subclass hooks --------------------------------------------------------

    def _get_column_headers(self) -> List[str]:
        raise NotImplementedError

    def _populate(self, data) -> None:
        raise NotImplementedError

    def _validate_row(self, row_data: List[str]) -> Optional[str]:
        raise NotImplementedError

    def _sync_changes(self, original, current) -> Optional[str]:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------

    def _add_row(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col in range(self._table.columnCount()):
            self._table.setItem(row, col, QTableWidgetItem(""))
        self._table.scrollToBottom()
        self._table.setCurrentCell(row, 0)
        self._table.editItem(self._table.item(row, 0))

    def _delete_rows(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()},
                      reverse=True)
        if not rows:
            return
        if len(rows) > 1 or self._table.item(rows[0], 0).text().strip():
            reply = QMessageBox.question(
                self, "Confirm Delete",
                f"Delete {len(rows)} row(s)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        for r in rows:
            self._table.removeRow(r)

    def _get_row_data(self, row: int) -> List[str]:
        return [
            (self._table.item(row, col).text().strip()
             if self._table.item(row, col) else "")
            for col in range(self._table.columnCount())
        ]

    def _on_accept(self) -> None:
        current_rows = []
        for row in range(self._table.rowCount()):
            data = self._get_row_data(row)
            # Skip completely blank rows silently
            if not any(data):
                continue
            err = self._validate_row(data)
            if err:
                QMessageBox.warning(self, "Validation Error",
                                    f"Row {row + 1}: {err}")
                return
            current_rows.append(data)

        err = self._sync_changes(self._original, current_rows)
        if err:
            QMessageBox.warning(self, "Error", err)
            return
        self.accept()


# ── XRAM Locals dialog ─────────────────────────────────────────────────────────

class LocalsTableDialog(_TableDialog):
    """
    Editable table of XRAM local variables for one function.

    Columns: Address | Type | Name
    """

    def __init__(self, func_ea: int, locals_list, parent=None):
        from pseudo8051.locals import get_locals
        try:
            import ida_funcs
            fn = ida_funcs.get_func(func_ea)
            fname = ida_funcs.get_func_name(fn.start_ea) if fn else hex(func_ea)
        except Exception:
            fname = hex(func_ea)

        self._func_ea = func_ea
        super().__init__(f"XRAM Locals — {fname}", parent)
        self._original = list(locals_list)   # snapshot for diffing
        self._populate(locals_list)

    def _get_column_headers(self) -> List[str]:
        return ["Address", "Type", "Name"]

    def _populate(self, data) -> None:
        self._table.setRowCount(0)
        for lv in data:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(f"{lv.addr:#06x}"))
            self._table.setItem(row, 1, QTableWidgetItem(lv.type))
            self._table.setItem(row, 2, QTableWidgetItem(lv.name))
        self._table.resizeColumnsToContents()

    def _validate_row(self, row_data: List[str]) -> Optional[str]:
        addr_s, type_s, name_s = row_data
        if not addr_s:
            return "Address is required."
        try:
            int(addr_s, 0)
        except ValueError:
            return f"Invalid address: {addr_s!r} (use hex like 0xdc8a or decimal)."
        if not type_s:
            return "Type is required."
        if not name_s:
            return "Name is required."
        return None

    def _sync_changes(self, original, current_rows) -> Optional[str]:
        from pseudo8051.locals import set_local, del_local

        # Map original by address
        orig_by_addr: Dict[int, object] = {lv.addr: lv for lv in original}

        # Build current set by address
        new_by_addr: Dict[int, Tuple[str, str]] = {}
        for addr_s, type_s, name_s in current_rows:
            try:
                addr = int(addr_s, 0)
            except ValueError:
                return f"Invalid address: {addr_s!r}"
            if addr in new_by_addr:
                return f"Duplicate address: {addr_s}"
            new_by_addr[addr] = (name_s, type_s)

        # Delete removed entries
        for old_addr in orig_by_addr:
            if old_addr not in new_by_addr:
                del_local(self._func_ea, old_addr)

        # Set new/changed entries
        for addr, (name, type_str) in new_by_addr.items():
            old = orig_by_addr.get(addr)
            if old is None or old.name != name or old.type != type_str:
                set_local(self._func_ea, addr, name, type_str)

        return None


# ── XRAM Parameters dialog ────────────────────────────────────────────────────

class XRAMParamsTableDialog(_TableDialog):
    """
    Editable table of XRAM parameters for one function.

    Columns: Address | Type | Name
    """

    def __init__(self, func_ea: int, params_list, parent=None):
        try:
            import ida_funcs
            fn = ida_funcs.get_func(func_ea)
            fname = ida_funcs.get_func_name(fn.start_ea) if fn else hex(func_ea)
        except Exception:
            fname = hex(func_ea)

        self._func_ea = func_ea
        super().__init__(f"XRAM Parameters — {fname}", parent)
        self._original = list(params_list)   # snapshot for diffing
        self._populate(params_list)

    def _get_column_headers(self) -> List[str]:
        return ["Address", "Type", "Name"]

    def _populate(self, data) -> None:
        self._table.setRowCount(0)
        for p in data:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(f"{p.addr:#06x}"))
            self._table.setItem(row, 1, QTableWidgetItem(p.type))
            self._table.setItem(row, 2, QTableWidgetItem(p.name))
        self._table.resizeColumnsToContents()

    def _validate_row(self, row_data: List[str]) -> Optional[str]:
        addr_s, type_s, name_s = row_data
        if not addr_s:
            return "Address is required."
        try:
            int(addr_s, 0)
        except ValueError:
            return f"Invalid address: {addr_s!r} (use hex like 0xdc8a or decimal)."
        if not type_s:
            return "Type is required."
        if not name_s:
            return "Name is required."
        return None

    def _sync_changes(self, original, current_rows) -> Optional[str]:
        from pseudo8051.xram_params import set_xram_param, del_xram_param

        orig_by_addr: Dict[int, object] = {p.addr: p for p in original}

        new_by_addr: Dict[int, Tuple[str, str]] = {}
        for addr_s, type_s, name_s in current_rows:
            try:
                addr = int(addr_s, 0)
            except ValueError:
                return f"Invalid address: {addr_s!r}"
            if addr in new_by_addr:
                return f"Duplicate address: {addr_s}"
            new_by_addr[addr] = (name_s, type_s)

        for old_addr in orig_by_addr:
            if old_addr not in new_by_addr:
                del_xram_param(self._func_ea, old_addr)

        for addr, (name, type_str) in new_by_addr.items():
            old = orig_by_addr.get(addr)
            if old is None or old.name != name or old.type != type_str:
                set_xram_param(self._func_ea, addr, name, type_str)

        return None


# ── Register Annotations dialog ────────────────────────────────────────────────

class RegAnnsTableDialog(_TableDialog):
    """
    Editable table of register annotations for one function.

    Columns: Address | Register(s) | Name | Type
    """

    def __init__(self, func_ea: int, reganns_list, parent=None):
        try:
            import ida_funcs
            fn = ida_funcs.get_func(func_ea)
            fname = ida_funcs.get_func_name(fn.start_ea) if fn else hex(func_ea)
        except Exception:
            fname = hex(func_ea)

        self._func_ea = func_ea
        super().__init__(f"Register Annotations — {fname}", parent)
        self._original = list(reganns_list)
        self._populate(reganns_list)

    def _get_column_headers(self) -> List[str]:
        return ["Address", "Register(s)", "Name", "Type"]

    def _populate(self, data) -> None:
        self._table.setRowCount(0)
        for ra in data:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(f"{ra.ea:#010x}"))
            self._table.setItem(row, 1, QTableWidgetItem(",".join(ra.regs)))
            self._table.setItem(row, 2, QTableWidgetItem(ra.name))
            self._table.setItem(row, 3, QTableWidgetItem(ra.type))
        self._table.resizeColumnsToContents()

    def _validate_row(self, row_data: List[str]) -> Optional[str]:
        addr_s, regs_s, name_s, type_s = row_data
        if not addr_s:
            return "Address is required."
        try:
            int(addr_s, 0)
        except ValueError:
            return f"Invalid address: {addr_s!r}"
        if not regs_s:
            return "Register(s) is required (e.g. R6 or R6,R7)."
        regs = [r.strip() for r in regs_s.split(',') if r.strip()]
        if not regs:
            return "No valid registers specified."
        for r in regs:
            if not r.startswith('R') and r not in ('A', 'B', 'DPTR', 'DPH', 'DPL'):
                return f"Unrecognised register: {r!r}"
        if not name_s:
            return "Name is required."
        if not type_s:
            return "Type is required."
        return None

    def _sync_changes(self, original, current_rows) -> Optional[str]:
        from pseudo8051.reganns import set_regann, del_regann

        # Map original by (ea, canonical_regs)
        orig_keys = {(ra.ea, ",".join(ra.regs)): ra for ra in original}

        # Validate for duplicates
        seen = set()
        new_entries = []
        for addr_s, regs_s, name_s, type_s in current_rows:
            try:
                ea = int(addr_s, 0)
            except ValueError:
                return f"Invalid address: {addr_s!r}"
            canon = ",".join(r.strip() for r in regs_s.split(',') if r.strip())
            key = (ea, canon)
            if key in seen:
                return f"Duplicate (address, register) pair: {addr_s} {regs_s}"
            seen.add(key)
            new_entries.append((ea, canon, name_s, type_s))

        new_keys = {(ea, canon) for ea, canon, _, _ in new_entries}

        # Delete removed
        for (ea, canon), ra in orig_keys.items():
            if (ea, canon) not in new_keys:
                del_regann(self._func_ea, ea, canon)

        # Set new/changed
        for ea, canon, name, type_str in new_entries:
            old = orig_keys.get((ea, canon))
            if old is None or old.name != name or old.type != type_str:
                set_regann(self._func_ea, ea, canon, name, type_str)

        return None
