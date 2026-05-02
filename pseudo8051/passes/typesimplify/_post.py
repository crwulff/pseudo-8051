"""
passes/typesimplify/_post.py — Post-simplify passes facade.

Re-exports everything from the individual sub-modules so that existing
`from pseudo8051.passes.typesimplify._post import ...` statements continue
to work without change.
"""

from pseudo8051.passes.typesimplify._dptr import (           # noqa: F401
    _is_dptr_inc_node,
    _collapse_dpl_dph,
    _collapse_dpl_dph_arithmetic,
    _dptr_live_after,
    _prune_orphaned_dptr_inc,
)
from pseudo8051.passes.typesimplify._xram_loads import (     # noqa: F401
    _consolidate_xram_local_loads,
    _subst_reg_in_scope,
)
from pseudo8051.passes.typesimplify._setup_fold import (     # noqa: F401
    _is_call_setup_assign,
    _collect_hir_name_refs,
    _fold_and_prune_setups,
    _fold_call_arg_pairs,
)
from pseudo8051.passes.typesimplify._return_fold import (    # noqa: F401
    _fold_return_chains,
)
from pseudo8051.passes.typesimplify._propagate import (      # noqa: F401
    _propagate_values,
    _fold_compound_assigns,
    _propagate_register_copies,
    _inline_retvals,
)
from pseudo8051.passes.typesimplify._carry import (          # noqa: F401
    _simplify_carry_comparison,
    _simplify_subb_jc,
    _simplify_cjne_jnc,
    _simplify_orl_zero_check,
    _simplify_arithmetic,
    _simplify_acc_bit_test,
)
from pseudo8051.passes.typesimplify._xram_call_args import ( # noqa: F401
    _fold_xram_call_args,
)


def _fold_inline_trampolines(hir):
    """
    Detect inlined cross-page trampoline patterns that survived as intra-function
    chunks (IDA owns them as part of the calling function, so resolve_callee is
    never invoked during lifting).

    Pattern 1 (DPTR-set + goto-stub at flat scope):
        Assign(Regs(DPTR), Const(offset))
        GotoStatement(label_Y)              ← label_Y matches *_call_page_N

    Pattern 2 (inlined code_call_page_N body, typically inside a structured node):
        A = IRAM[current_code_page] ...     ← optional
        push(page_byte)
        push(ret_stub_addr)
        push(DPL)
        push(DPH)                           ← annotation carries DPTR = func_name
        GotoStatement(label_Z)             ← label_Z is a page-epilogue block

    Replace with ExprStmt(Call(real_target)) and remove the now-dead page-switch
    helper body.  Recurses into structured bodies first.
    """
    import re
    from pseudo8051.ir.hir import Assign, GotoStatement, Label, ExprStmt, ReturnStmt
    from pseudo8051.ir.expr import Regs, Const, Call, IRAMRef, XRAMRef
    from pseudo8051.constants import dbg

    # ── helpers ──────────────────────────────────────────────────────────────

    def _dptr_offset(node):
        """Return the DPTR constant offset if node is Assign(DPTR..., Const), else None."""
        if not isinstance(node, Assign):
            return None
        if not isinstance(node.rhs, Const):
            return None
        lhs = node.lhs
        if isinstance(lhs, Regs) and set(lhs.names) <= {'DPTR', 'DPL', 'DPH'}:
            return node.rhs.value & 0xFFFF
        return None

    def _collect_goto_labels(nodes):
        """Collect all GotoStatement target labels in a flat node list."""
        targets = set()
        for nd in nodes:
            if isinstance(nd, GotoStatement):
                targets.add(nd.label)
        return targets

    def _is_push(node, reg=None):
        """True if node is ExprStmt(Call('push', [Regs((reg,))])) (or any push if reg is None)."""
        if not isinstance(node, ExprStmt):
            return False
        expr = node.expr
        if not (isinstance(expr, Call) and expr.func == 'push' and len(expr.args) == 1):
            return False
        if reg is None:
            return True
        arg = expr.args[0]
        return isinstance(arg, Regs) and arg.names == (reg,)

    def _is_iram_assign(node):
        """True if node is Assign(Regs('A'), expr-involving-IRAMRef)."""
        if not (isinstance(node, Assign)
                and isinstance(node.lhs, Regs) and node.lhs.names == ('A',)):
            return False
        # Check if rhs touches IRAM (IRAMRef anywhere in the expression tree)
        def _has_iram(e):
            if isinstance(e, IRAMRef):
                return True
            return any(_has_iram(c) for c in e.children())
        return _has_iram(node.rhs)

    def _is_page_epilogue_block(nodes, start):
        """
        Return True if nodes[start:] begins a page-restore block:
        optional Label nodes, then IRAM assign + XRAM assign + ReturnStmt.
        """
        k = start
        while k < len(nodes) and isinstance(nodes[k], Label):
            k += 1
        has_iram = has_xram = has_ret = False
        while k < len(nodes) and not isinstance(nodes[k], Label):
            nd = nodes[k]
            if isinstance(nd, Assign) and isinstance(nd.lhs, IRAMRef):
                has_iram = True
            elif isinstance(nd, Assign) and isinstance(nd.lhs, XRAMRef):
                has_xram = True
            elif isinstance(nd, ReturnStmt):
                has_ret = True
                break
            k += 1
        return has_ret and (has_iram or has_xram)

    def _find_page_epilogue_labels(nodes):
        """Collect all labels of page-restore epilogue blocks in a flat node list."""
        labels = set()
        i = 0
        while i < len(nodes):
            if isinstance(nodes[i], Label):
                if _is_page_epilogue_block(nodes, i):
                    labels.add(nodes[i].name)
            i += 1
        return labels

    def _dptr_func_name(nodes, around_idx):
        """Scan nearby nodes for DPTR = Name(func_name) in reg_exprs."""
        from pseudo8051.ir.expr import Name as _NameExpr
        for k in range(around_idx, max(-1, around_idx - 12), -1):
            ann = getattr(nodes[k], 'ann', None)
            if ann and ann.reg_exprs:
                dptr_val = ann.reg_exprs.get('DPTR')
                if isinstance(dptr_val, _NameExpr):
                    return dptr_val.name
        return None

    def _build_call_node(ea, real_name):
        """Build ExprStmt(Call(real_name, args)) using prototype if available."""
        try:
            from pseudo8051.prototypes import get_proto, param_regs
            from pseudo8051.ir.expr import RegGroup, Name as _Name
            _proto = get_proto(real_name)
            if _proto:
                _p_regs = param_regs(_proto)
                _args = []
                for _p, _regs in zip(_proto.params, _p_regs):
                    if _regs:
                        _args.append(RegGroup(_regs) if len(_regs) > 1 else _Name("".join(_regs)))
                    else:
                        _args.append(_Name(_p.name))
                return ExprStmt(ea, Call(real_name, _args))
        except Exception:
            pass
        return ExprStmt(ea, Call(real_name, []))

    def _fold_preamble_in_list(nodes, epilogue_labels):
        """
        Replace inlined call_page_N preambles in a flat node list.

        Pattern (ends with push(DPL); push(DPH); goto epilogue_label):
          [A = IRAM[page] ...] push(page_byte) push(ret_addr) push(DPL) push(DPH) goto E
        Replace everything from the first push backward (including optional A=IRAM) with
        ExprStmt(Call(func)), where func comes from DPTR annotation on the push nodes.
        """
        result = list(nodes)
        i = 0
        while i < len(result):
            # Match: goto epilogue_label
            if not (isinstance(result[i], GotoStatement)
                    and result[i].label in epilogue_labels):
                i += 1
                continue
            # Verify push(DPH) and push(DPL) immediately precede the goto
            if i < 2 or not (_is_push(result[i-1], 'DPH')
                              and _is_push(result[i-2], 'DPL')):
                i += 1
                continue
            # Get the function name from DPTR annotation on nearby nodes
            func_name = _dptr_func_name(result, i - 1)
            if func_name is None:
                i += 1
                continue
            dbg("trampolines",
                f"  fold-preamble @{hex(result[i].ea)}: goto {result[i].label} → {func_name}")
            # Find preamble start: scan backward past consecutive pushes and
            # optional A=IRAM[page] assignment.
            preamble_start = i - 2  # push(DPL)
            while preamble_start > 0 and _is_push(result[preamble_start - 1]):
                preamble_start -= 1
            if (preamble_start > 0
                    and _is_iram_assign(result[preamble_start - 1])):
                preamble_start -= 1
            call_node = _build_call_node(result[preamble_start].ea, func_name)
            # Replace preamble_start..i (inclusive) with the call
            result = result[:preamble_start] + [call_node] + result[i + 1:]
            i = preamble_start + 1
        return result

    # ── Phase 1: recurse into structured bodies ───────────────────────────────
    # Must happen before flat-level fold so that page-epilogue labels can be
    # identified from the still-present flat-level blocks.
    epilogue_labels = _find_page_epilogue_labels(hir)
    if epilogue_labels:
        dbg("trampolines", f"  page-epilogue labels: {sorted(epilogue_labels)}")

    def _recurse_and_fold_preamble(nodes):
        nodes = [n.map_bodies(_recurse_and_fold_preamble) for n in nodes]
        if epilogue_labels:
            nodes = _fold_preamble_in_list(nodes, epilogue_labels)
        return nodes

    hir = _recurse_and_fold_preamble(hir)

    # ── Phase 2: flat-level DPTR=Const + goto *_call_page_N fold ─────────────
    result = list(hir)
    i = 0
    while i < len(result):
        offset = _dptr_offset(result[i])
        if offset is None:
            i += 1
            continue

        # Find next non-Label node
        j = i + 1
        while j < len(result) and isinstance(result[j], Label):
            j += 1
        if j >= len(result) or not isinstance(result[j], GotoStatement):
            i += 1
            continue

        target_label = result[j].label
        # Detect page-switch helper by name pattern: *_call_page_N
        m = re.match(r'.*_call_page_(\d+)$', target_label)
        if not m:
            i += 1
            continue
        page_num = int(m.group(1))

        try:
            import ida_name, ida_funcs
            from pseudo8051.trampolines import _page_segment_base
            real_ea = _page_segment_base(page_num) | offset
            real_name = (ida_funcs.get_func_name(real_ea)
                         or ida_name.get_name(real_ea)
                         or hex(real_ea))
        except Exception:
            i += 1
            continue

        dbg("trampolines",
            f"  fold-inline-trampoline @{hex(result[i].ea)}: "
            f"DPTR={hex(offset)} → page {page_num} → {real_name}")

        dptr_node  = result[i]
        goto_node  = result[j]
        new_node = _build_call_node(result[i].ea, real_name)
        new_node.source_nodes = [dptr_node, goto_node]
        # Replace result[i] (DPTR assign) with new_node; remove result[j] (goto)
        result = result[:i] + [new_node] + result[i + 1:j] + result[j + 1:]
        # result[i] = new call node; dead page-switch body starts at i+1

        # Remove the now-dead page-switch helper body.
        # Collect goto targets from the live portion (everything up to and
        # including the inserted call) so we know which labels are still needed.
        live_refs = _collect_goto_labels(result[:i + 1])

        k = i + 1
        while k < len(result):
            nd = result[k]
            # Stop at a Label that is referenced from outside the dead zone
            if isinstance(nd, Label) and nd.name in live_refs:
                break
            was_return = isinstance(nd, ReturnStmt)
            k += 1
            if was_return:
                break  # page-switch chain ends at its RET

        if k > i + 1:
            result = result[:i + 1] + result[k:]

        i += 1

    # ── Phase 3: strip page-restore assignments from surviving epilogue blocks ─
    # After preamble folding, page-epilogue blocks (IRAM[page]=0; XRAM[...]=0;
    # return) that remain in the flat HIR are pure trampoline boilerplate.
    # Keep only their Labels and ReturnStmt; remove the page-restore Assigns.
    if epilogue_labels:
        cleaned = []
        i = 0
        while i < len(result):
            nd = result[i]
            if isinstance(nd, Label) and nd.name in epilogue_labels:
                # Emit the label(s), then skip non-ReturnStmt/non-Label nodes
                cleaned.append(nd)
                i += 1
                while i < len(result) and not isinstance(result[i], (Label, ReturnStmt)):
                    i += 1   # drop page-restore Assigns
            else:
                cleaned.append(nd)
                i += 1
        result = cleaned

    return result


def _rename_byte_field_lhs(hir, byte_field_by_reg: dict):
    """
    Rename raw register LHS to their byte-field Name equivalents.

    After TypeAwareSimplifier substitutes RHS reads (e.g. R1 → src.lo) but leaves
    LHS registers unchanged, this produces mismatched nodes like:
        R1 = src.lo + expr;

    This pass renames the LHS when the register backs a byte-field variable:
        src.lo = src.lo + expr;

    This allows collapse_mb_assigns to subsequently collapse hi/lo field pairs.
    Only runs when byte_field_by_reg is non-empty (i.e. the function has 16-bit
    parameters whose byte-field VarInfo entries were recorded before _simplify).
    """
    if not byte_field_by_reg:
        return hir

    from pseudo8051.ir.hir import Assign
    from pseudo8051.ir.expr import Regs as RegExpr, Name as NameExpr

    def _rename(nodes):
        result = []
        for node in nodes:
            mapped = node.map_bodies(_rename)
            if mapped is not node:
                result.append(mapped)
                continue
            if (isinstance(node, Assign)
                    and isinstance(node.lhs, RegExpr)
                    and node.lhs.is_single):
                vi = byte_field_by_reg.get(node.lhs.name)
                if vi is not None and vi.name:
                    new_node = Assign(node.ea, NameExpr(vi.name), node.rhs)
                    node.copy_meta_to(new_node)
                    new_node.source_nodes = [node]
                    result.append(new_node)
                    continue
            result.append(mapped)
        return result

    return _rename(hir)


def recurse_bodies(nodes, fn):
    """Recurse fn into structured node bodies; pass other nodes through."""
    return [node.map_bodies(fn) for node in nodes]


def _subst_xram_in_hir(nodes, reg_map):
    """
    Walk the HIR and apply _subst_xram_in_expr to every expression.

    Called after _collapse_dpl_dph_arithmetic to convert newly-created
    XRAM[base + idx] nodes into array subscript expressions (e.g. foo[R6R7]).
    Also applied to LHS write positions (XRAMRef indirect writes) so that
    indexed writes like XRAM[base + idx] = ... become arr[idx] = ... .
    """
    from pseudo8051.passes.patterns._utils import _subst_xram_in_expr, _apply_expr_subst_to_node
    from pseudo8051.passes.patterns._node_utils import _fold_exprs_in_node
    from pseudo8051.ir.hir import Assign
    from pseudo8051.ir.expr import XRAMRef

    def _subst_fn(e):
        return _subst_xram_in_expr(e, reg_map)

    def _visit(ns):
        return _subst_xram_in_hir(ns, reg_map)

    result = []
    for node in nodes:
        patched = _apply_expr_subst_to_node(node, _subst_fn)
        # Also transform XRAMRef LHS (write destination) so indexed XRAM writes
        # like XRAM[base + idx] = expr become arr[idx] = expr.
        if isinstance(patched, Assign) and isinstance(patched.lhs, XRAMRef):
            new_lhs = _subst_fn(patched.lhs)
            if new_lhs is not patched.lhs:
                prev = patched
                patched = patched.copy_meta_to(Assign(patched.ea, new_lhs, patched.rhs))
                patched.source_nodes = [prev]
        # Fold after substitution so that byte-field reconstructions like
        # (x.hi << 8) | x.lo collapse to Name("x").
        if patched is not node:
            patched = _fold_exprs_in_node(patched)
        result.append(patched.map_bodies(_visit))
    return result


def _subst_iram_in_hir(nodes, reg_map):
    """Walk the HIR and apply _subst_iram_in_expr to every expression.

    Handles both RHS reads and IRAMRef LHS write positions so that
    IRAM[addr] = expr becomes local_name = expr as well.
    """
    from pseudo8051.passes.patterns._utils import _subst_iram_in_expr, _apply_expr_subst_to_node
    from pseudo8051.ir.hir import Assign
    from pseudo8051.ir.expr import IRAMRef

    def _subst_fn(e):
        return _subst_iram_in_expr(e, reg_map)

    def _visit(ns):
        return _subst_iram_in_hir(ns, reg_map)

    result = []
    for node in nodes:
        patched = _apply_expr_subst_to_node(node, _subst_fn)
        if isinstance(patched, Assign) and isinstance(patched.lhs, IRAMRef):
            new_lhs = _subst_fn(patched.lhs)
            if new_lhs is not patched.lhs:
                prev = patched
                patched = patched.copy_meta_to(Assign(patched.ea, new_lhs, patched.rhs))
                patched.source_nodes = [prev]
        result.append(patched.map_bodies(_visit))
    return result
