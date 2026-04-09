"""
passes/debug_dump.py — HIR dump utility for pass-level debug output.

Exports:
  dump_pass_hir(pass_name, hir_list, func_name)   write /tmp/<pass>.hir
"""

import os
from typing import List

_DBG_DIR = "/tmp/pseudo8051"


def _collect_line_chains(nodes: list, node_map: dict,
                          offset: int, ancestors: tuple) -> int:
    """Map every rendered-line index to its full HIR containment chain (deepest last).

    Mirrors pseudo8051.__init__._collect_line_chains but adds DoWhileNode.
    """
    for node in nodes:
        node_lines = node.render(indent=0)
        chain = ancestors + (node,)
        for k in range(len(node_lines)):
            node_map[offset + k] = chain
        name = type(node).__name__
        sub = offset + 1
        if name == 'IfNode':
            sub = _collect_line_chains(node.then_nodes, node_map, sub, chain)
            if node.else_nodes:
                sub += 1   # "} else {"
                sub = _collect_line_chains(node.else_nodes, node_map, sub, chain)
        elif name in ('WhileNode', 'ForNode', 'DoWhileNode'):
            _collect_line_chains(node.body_nodes, node_map, sub, chain)
        # SwitchNode: no child HIR lists to recurse into
        offset += len(node_lines)
    return offset


def dump_pass_hir(pass_name: str, hir_list: list, func_name: str = "") -> None:
    """Write annotated flat HIR to /tmp/<pass_name>.hir, overwriting any previous file.

    Format mirrors the viewer's annotation display: each rendered C line is
    followed by ``    // <ann>`` lines from node.ann_lines() + node.node_ann_lines()
    of the deepest enclosing node at that line position.
    """
    from pseudo8051.constants import DEBUG
    if not DEBUG:
        return

    # Build line-index → deepest-node chain map
    node_map: dict = {}
    _collect_line_chains(hir_list, node_map, 0, ())

    # Flatten all rendered lines
    all_lines: List[tuple] = []
    for node in hir_list:
        all_lines.extend(node.render(indent=1))

    os.makedirs(_DBG_DIR, exist_ok=True)
    path = os.path.join(_DBG_DIR, f"{pass_name}.hir")
    with open(path, 'w') as f:
        if func_name:
            f.write(f"// ── {pass_name}: {func_name} ──\n")
        for i, (ea, text) in enumerate(all_lines):
            f.write(text + '\n')
            chain = node_map.get(i)
            if chain:
                node = chain[-1]
                anns = node.ann_lines() + node.node_ann_lines()
                if anns:
                    anns[0] = f"{anns[0]} [{hex(node.ea)}]"
                for ann in anns:
                    f.write(f"    // {ann}\n")
