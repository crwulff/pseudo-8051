"""
pseudo8051/detail_viewer.py — DetailViewer: per-line traceability pane.

Shows the assembly instruction(s) that generated the selected pseudocode line,
with optional HIR annotations and configurable depth into child HIR nodes.

Open via right-click → "Show detail pane" in any pseudocode viewer tab.
The detail tab auto-updates whenever the cursor moves in a pseudocode viewer.

Each pseudocode tab ("8051 Pseudocode-A") gets its own detail tab
("8051 Detail-A") so multiple functions can be open simultaneously.

Node map indexing note: _node_map is keyed by the raw func.render() line index
(starting at offset 2 for the first HIR node, past the signature and '{').
When the pseudocode viewer is NOT in "annotate" mode (the default), the raw
index equals the viewer line number, so GetLineNo() works as a direct key.
"""

import sys
from typing import List, Optional, Tuple

import ida_kernwin
import idc

from pseudo8051.ir.hir import (HIRNode, IfNode, WhileNode, ForNode,
                                DoWhileNode, SwitchNode)

# ── Module-level viewer registry (reset on each reload) ───────────────────────

_BASE_DETAIL_TITLE = "8051 Detail"
_BASE_PV_TITLE     = "8051 Pseudocode"

# keyed by detail tab title (e.g. "8051 Detail-A")
_viewers: dict = {}

# The detail viewer most recently right-clicked; used by action handlers.
_active_viewer: Optional["DetailViewer"] = None


# ── HIR child helpers ─────────────────────────────────────────────────────────

def _get_hir_children(node) -> List[HIRNode]:
    """Direct HIR child nodes of a container node; empty list for leaves."""
    # _SwitchCaseView sentinel: return just this case arm's body nodes
    if hasattr(node, 'case_body'):
        return list(node.case_body())
    if isinstance(node, IfNode):
        return list(node.then_nodes) + list(node.else_nodes)
    if isinstance(node, (WhileNode, ForNode, DoWhileNode)):
        return list(node.body_nodes)
    if isinstance(node, SwitchNode):
        children: List[HIRNode] = []
        for _, body in node.cases:
            if isinstance(body, list):
                children.extend(body)
        if isinstance(node.default_body, list):
            children.extend(node.default_body)
        return children
    return []


def _collect_nodes_with_depth(
        node: HIRNode, max_depth: int) -> List[Tuple[int, HIRNode]]:
    """
    BFS over HIR children, yielding (relative_depth, child) pairs.
    max_depth=0 → empty (children hidden);
    max_depth=1 → direct children only;
    max_depth=-1 → unlimited depth.
    """
    if max_depth == 0:
        return []
    result: List[Tuple[int, HIRNode]] = []
    queue: List[Tuple[int, HIRNode]] = [
        (1, c) for c in _get_hir_children(node)]
    while queue:
        depth, child = queue.pop(0)
        result.append((depth, child))
        if max_depth == -1 or depth < max_depth:
            for grandchild in _get_hir_children(child):
                queue.append((depth + 1, grandchild))
    return result


def _sorted_eas(node: HIRNode, max_depth: int) -> List[int]:
    """EAs from node's source_nodes tree in execution order (post-order DFS, unique).

    Post-order (children before self) matches execution order because each
    synthetic/merged node's sources were executed before the node itself.
    max_depth is unused but kept for API compatibility.
    """
    seen: set = set()
    result: List[int] = []

    def _walk(n: object) -> None:
        for sn in getattr(n, 'source_nodes', []):
            _walk(sn)
        ea = getattr(n, 'ea', None)
        if ea is not None and ea not in seen:
            seen.add(ea)
            result.append(ea)

    _walk(node)
    return result


def _collect_source_tree(node: HIRNode) -> List[Tuple[int, HIRNode]]:
    """DFS over source_nodes, yielding (depth, node) pairs (depth 0 = direct sources)."""
    result: List[Tuple[int, HIRNode]] = []
    def _walk(n: HIRNode, d: int) -> None:
        for sn in getattr(n, 'source_nodes', []):
            result.append((d, sn))
            _walk(sn, d + 1)
    _walk(node, 0)
    return result


# ── Detail viewer ─────────────────────────────────────────────────────────────

_SEP = "─" * 52


class DetailViewer(ida_kernwin.simplecustviewer_t):
    """
    Dockable tab showing traceability for the selected pseudocode line:
      • source instruction address(es) and disassembly text
      • optional HIR node annotations (same as "Annotate HIR" mode)
      • optional child HIR nodes at configurable depth
    """

    def Create(self, title: str, pv_title: str) -> bool:
        self.show_annotations: bool = False
        self.show_sources:     bool = False
        self.child_depth:      int  = 0   # 0=hidden, 1,2,...=N levels, -1=all
        self._visible:         bool = False
        self._title:           str  = title
        self._pv_title:        str  = pv_title   # matching pseudocode tab title
        self._line_ea_map:     dict = {}          # line_no → ea for instruction lines
        return super().Create(title)

    # ── Content rendering ─────────────────────────────────────────────────

    def _add_insn_line(self, ea: int, text: str) -> None:
        """Add an instruction line and record the EA for double-click navigation."""
        line_no = self.Count()
        self._line_ea_map[line_no] = ea
        self.AddLine(text)

    def update_from(self, pseudocode_viewer) -> None:
        """Refresh detail content from the pseudocode viewer's current line."""
        self.ClearLines()
        self._line_ea_map = {}

        line_no  = pseudocode_viewer.GetLineNo()
        node_map = getattr(pseudocode_viewer, '_node_map', {})
        chain    = node_map.get(line_no)

        if chain is None:
            self.AddLine("No HIR node for this line.")
            self.Refresh()
            return

        node = chain[-1]

        # _SwitchCaseView sentinel: case/default header line
        is_case_view = hasattr(node, 'case_body')

        # Header
        if is_case_view:
            line_text = node.case_label()
            self.AddLine(f"Line {line_no}:  {line_text}")
            self.AddLine(f"Node: SwitchNode  EA: {hex(node.ea)}")
        else:
            first_render = node.render(indent=0)
            line_text    = first_render[0][1].strip() if first_render else ""
            self.AddLine(f"Line {line_no}:  {line_text}")
            self.AddLine(f"Node: {type(node).__name__}  EA: {hex(node.ea)}")
        self.AddLine(_SEP)

        # Instructions for this node (src_eas present on both HIRNode and _SwitchCaseView)
        self.AddLine("Instructions:")
        for ea in _sorted_eas(node, 0):
            disasm = idc.GetDisasm(ea)
            self._add_insn_line(ea, f"  {hex(ea)}  {disasm if disasm else '(no disasm)'}")

        # Source node provenance tree
        self.AddLine("")
        if not self.show_sources:
            self.AddLine("Source nodes (hidden)")
        else:
            self.AddLine("Source nodes:")
            source_tree = _collect_source_tree(node)
            if not source_tree:
                self.AddLine("  [no source nodes — leaf from IDA]")
            else:
                for depth, sn in source_tree:
                    pad   = "  " * (depth + 1)
                    stype = type(sn).__name__
                    first = sn.render(indent=0)
                    stext = first[0][1].strip() if first else ""
                    self.AddLine(f"{pad}[depth={depth}]  {stype}  EA: {hex(sn.ea)}")
                    self._add_insn_line(sn.ea, f"{pad}  {stext!r}")

        # Optional annotations
        if self.show_annotations:
            ann_node = node.switch_node if is_case_view else node
            ann_lines = ann_node.ann_lines() + ann_node.node_ann_lines()
            if ann_lines:
                self.AddLine("")
                self.AddLine("Expressions:")
                for a in ann_lines:
                    self.AddLine(f"  // {a}")

        # Children section
        self.AddLine("")
        if self.child_depth == 0:
            self.AddLine("Children (depth 0 — hidden)")
        else:
            depth_label = str(self.child_depth) if self.child_depth > 0 else "all"
            self.AddLine(f"Children (depth {depth_label}):")
            children = _collect_nodes_with_depth(node, self.child_depth)
            if not children:
                self.AddLine("  [no HIR children — leaf node]")
            else:
                for depth, child in children:
                    pad       = "  " * depth
                    ctype     = type(child).__name__
                    self.AddLine(
                        f"{pad}[depth={depth}]  {ctype}  EA: {hex(child.ea)}")
                    for cea in _sorted_eas(child, 0):
                        cdisasm = idc.GetDisasm(cea)
                        self._add_insn_line(
                            cea,
                            f"{pad}  {hex(cea)}  "
                            f"{cdisasm if cdisasm else '(no disasm)'}")
                    if self.show_annotations:
                        for a in child.ann_lines() + child.node_ann_lines():
                            self.AddLine(f"{pad}  // {a}")

        self.Refresh()

    # ── Double-click navigation ───────────────────────────────────────────

    def OnDblClick(self, _shift: int) -> bool:
        """Jump IDA disassembly view to the address of the double-clicked line."""
        ea = self._line_ea_map.get(self.GetLineNo())
        if ea is not None:
            idc.jumpto(ea)
        return True

    # ── Context menu ──────────────────────────────────────────────────────

    def OnPopup(self, form, popup_handle) -> bool:
        global _active_viewer
        _active_viewer = self   # track which viewer the user right-clicked

        ann_label = ("Hide expressions" if self.show_annotations
                     else "Show expressions")
        ida_kernwin.update_action_label(
            "pseudo8051:detail_toggle_ann", ann_label)
        ida_kernwin.attach_action_to_popup(
            form, popup_handle, "pseudo8051:detail_toggle_ann", "")

        src_label = ("Hide source nodes" if self.show_sources
                     else "Show source nodes")
        ida_kernwin.update_action_label(
            "pseudo8051:detail_toggle_sources", src_label)
        ida_kernwin.attach_action_to_popup(
            form, popup_handle, "pseudo8051:detail_toggle_sources", "")

        for action_id, label in [
            ("pseudo8051:detail_depth_0",   "Children: off"),
            ("pseudo8051:detail_depth_1",   "Children: depth 1"),
            ("pseudo8051:detail_depth_2",   "Children: depth 2"),
            ("pseudo8051:detail_depth_all", "Children: all"),
        ]:
            ida_kernwin.attach_action_to_popup(
                form, popup_handle, action_id, "Children/")
        return False

    def OnClose(self) -> None:
        global _active_viewer
        _viewers.pop(self._title, None)
        if _active_viewer is self:
            _active_viewer = None
        self._visible = False


# ── Action handlers ───────────────────────────────────────────────────────────

def _refresh_active() -> None:
    """Re-run update_from() using _active_viewer's paired pseudocode viewer."""
    if _active_viewer is None or not _active_viewer._visible:
        return
    pv_title = _active_viewer._pv_title
    pv = getattr(sys, '_pseudo8051_viewers', {}).get(pv_title)
    if pv is not None and ida_kernwin.find_widget(pv_title) is not None:
        _active_viewer.update_from(pv)


class _ToggleAnnAction(ida_kernwin.action_handler_t):
    def activate(self, ctx) -> int:
        if _active_viewer is not None:
            _active_viewer.show_annotations = not _active_viewer.show_annotations
            _refresh_active()
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


class _ToggleSourcesAction(ida_kernwin.action_handler_t):
    def activate(self, ctx) -> int:
        if _active_viewer is not None:
            _active_viewer.show_sources = not _active_viewer.show_sources
            _refresh_active()
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


class _SetDepthAction(ida_kernwin.action_handler_t):
    def __init__(self, depth: int) -> None:
        super().__init__()
        self._depth = depth

    def activate(self, ctx) -> int:
        if _active_viewer is not None:
            _active_viewer.child_depth = self._depth
            _refresh_active()
        return 1

    def update(self, ctx) -> int:
        return ida_kernwin.AST_ENABLE_ALWAYS


def _register_detail_actions() -> None:
    _defs = [
        ("pseudo8051:detail_toggle_ann",     "Show expressions",    _ToggleAnnAction()),
        ("pseudo8051:detail_toggle_sources", "Show source nodes",   _ToggleSourcesAction()),
        ("pseudo8051:detail_depth_0",        "Children: off",       _SetDepthAction(0)),
        ("pseudo8051:detail_depth_1",        "Children: depth 1",   _SetDepthAction(1)),
        ("pseudo8051:detail_depth_2",        "Children: depth 2",   _SetDepthAction(2)),
        ("pseudo8051:detail_depth_all",      "Children: all",       _SetDepthAction(-1)),
    ]
    for name, label, handler in _defs:
        ida_kernwin.unregister_action(name)
        ida_kernwin.register_action(
            ida_kernwin.action_desc_t(name, label, handler))


# ── Public entry point ────────────────────────────────────────────────────────

def show_detail_viewer(pseudocode_viewer) -> None:
    """
    Open (or bring into focus) the detail viewer paired with pseudocode_viewer.

    The detail tab title mirrors the pseudocode tab suffix so IDA treats them
    as distinct windows (e.g. "8051 Pseudocode-A" → "8051 Detail-A").
    Subsequent updates are driven by OnClick / OnKeydown in PseudocodeViewer.
    """
    global _active_viewer

    pv_title = getattr(pseudocode_viewer, '_title', '')
    suffix   = pv_title[len(_BASE_PV_TITLE):]   # e.g. "-A"
    detail_title = f"{_BASE_DETAIL_TITLE}{suffix}"

    # Reuse the existing viewer for this tab if it is still open.
    viewer = _viewers.get(detail_title)
    if viewer is None or not viewer._visible:
        viewer = DetailViewer()
        if not viewer.Create(detail_title, pv_title):
            return
        _viewers[detail_title] = viewer

    _active_viewer = viewer

    # Populate content before displaying
    viewer.update_from(pseudocode_viewer)

    # Dock next to the source pseudocode tab
    _DP_TAB      = getattr(ida_kernwin, 'DP_TAB',      6)
    _WOPN_DP_TAB = getattr(ida_kernwin, 'WOPN_DP_TAB', _DP_TAB << 16)
    _DP_INSIDE   = getattr(ida_kernwin, 'DP_INSIDE',   5)

    dest      = "IDA View-A"
    pv_widget = pseudocode_viewer.GetWidget()
    if pv_widget is not None:
        dest = ida_kernwin.get_widget_title(pv_widget) or dest

    w = viewer.GetWidget()
    if w is not None:
        try:
            ida_kernwin.display_widget(w, _WOPN_DP_TAB, dest)
        except TypeError:
            ida_kernwin.set_dock_pos(detail_title, dest, _DP_INSIDE)
            ida_kernwin.display_widget(w, 0)

    viewer._visible = True


_register_detail_actions()
