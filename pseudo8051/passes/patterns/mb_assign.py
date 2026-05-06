"""
passes/patterns/mb_assign.py — collapse_mb_assigns().

Collapses consecutive byte-field Assign nodes produced after per-byte
XRAM-write pattern processing:

    Assign(ea, Name("var0.hi"), Name("count.hi"))
    DPTR++  (ExprStmt — skipped)
    Assign(ea, Name("var0.lo"), Name("count.lo"))

into a single assignment:

    Assign(ea, Name("var0"), Name("count"))

Also handles constant sources:

    Assign(ea, Name("var0.hi"), Name("0x12"))  or Const(0x12)
    DPTR++
    Assign(ea, Name("var0.lo"), Name("0x34"))  or Const(0x34)
  →
    Assign(ea, Name("var0"), Const(0x1234))

This must run as a SECOND PASS on the already-simplified HIR output, because
the byte-field Assign nodes are only visible after the first simplification
round (XRAMLocalWritePattern + AccumRelayPattern have already transformed the
raw XRAM-write nodes into named Assign nodes).
"""

import re
from typing import List, Optional, Tuple

from pseudo8051.ir.hir   import HIRNode, Assign, TypedAssign, CompoundAssign, ExprStmt
from pseudo8051.ir.expr  import Reg, Regs, UnaryOp, Name, Const, BinOp, Cast
from pseudo8051.constants import dbg

# Match a bare integer constant (used to parse Name("0x12") from XRAMLocalWritePattern)
_RE_CONST_VAL = re.compile(r'^(0x[0-9a-fA-F]+|\d+)$')


def _parse_int(s: str) -> Optional[int]:
    m = _RE_CONST_VAL.match(s.strip())
    if not m:
        return None
    t = m.group(1)
    return int(t, 16) if t.lower().startswith("0x") else int(t)


def _split_field(name: str) -> Optional[Tuple[str, str]]:
    """Split 'parent.suffix' into (parent, suffix), or return None."""
    if "." in name:
        parent, suffix = name.split(".", 1)
        return (parent, suffix)
    return None


def _is_skippable(node: HIRNode) -> bool:
    """True if node can be skipped between byte-field assign pairs.

    Skips DPTR++ increments (normal XRAM traversal side-effect) and bare
    ExprStmt(Const(...)) nodes (residual XRAM-address constants left over when
    the surrounding movx pattern wasn't fully folded).
    """
    if isinstance(node, ExprStmt):
        if isinstance(node.expr, UnaryOp):
            return node.expr.op == "++" and node.expr.operand == Reg("DPTR")
        if isinstance(node.expr, Const):
            return True
    return False


def _valid_suffix_sequence(suffixes: List[str]) -> bool:
    """True if suffixes form a valid ordered byte-field sequence."""
    if suffixes == ["hi", "lo"] or suffixes == ["lo", "hi"]:
        return True
    return suffixes == [f"b{i}" for i in range(len(suffixes))]


def _rhs_as_const(rhs) -> Optional[int]:
    """Return integer value if rhs is a Const or a Name holding a numeric literal."""
    if isinstance(rhs, Const):
        return rhs.value
    if isinstance(rhs, Name):
        return _parse_int(rhs.name)
    return None


def _try_collapse_rmw_inc(nodes: List[HIRNode], i: int):
    """
    Recognize the accumulator read-modify-write increment pattern for XRAM locals:

      A = <var>;       (load XRAM local into A)
      <var> = ++A;     (increment A and store back)

    Collapses to a post-increment statement:
      <var>++;
    """
    n0 = nodes[i]
    if not (isinstance(n0, Assign)
            and isinstance(n0.lhs, Regs) and n0.lhs.is_single
            and n0.lhs.names == ('A',)
            and isinstance(n0.rhs, Name)):
        return None, i

    var_name = n0.rhs.name
    if i + 1 >= len(nodes):
        return None, i

    n1 = nodes[i + 1]
    if not (isinstance(n1, Assign)
            and isinstance(n1.lhs, Name)
            and n1.lhs.name == var_name):
        return None, i

    rhs = n1.rhs
    is_inc_a = (
        (isinstance(rhs, Name) and rhs.name == '++A')
        or (isinstance(rhs, UnaryOp) and rhs.op == '++'
            and not getattr(rhs, 'post', True)
            and isinstance(rhs.operand, Regs)
            and rhs.operand.names == ('A',))
    )
    if not is_inc_a:
        return None, i

    dbg("typesimp", f"  mb_assign: {var_name}++ (rmw-inc)")
    result = ExprStmt(n0.ea, UnaryOp('++', Name(var_name), post=True))
    result.source_nodes = [n0, n1]
    return result, i + 2


def _try_collapse(nodes: List[HIRNode], i: int):
    """
    Attempt to collapse a byte-field assignment sequence starting at nodes[i].

    Returns (collapsed_node, new_i) on success, or (None, i) if no collapse.
    """
    n0 = nodes[i]
    if not isinstance(n0, Assign) or not isinstance(n0.lhs, Name):
        return None, i

    field0 = _split_field(n0.lhs.name)
    if field0 is None or field0[1] not in ("hi", "lo", "b0"):
        return None, i

    lhs_parent, first_suffix = field0

    # Scan forward collecting (suffix, rhs_expr) pairs, skipping DPTR++ nodes
    pairs: List[Tuple[str, object]] = [(first_suffix, n0.rhs)]
    j = i + 1
    while j < len(nodes):
        nd = nodes[j]
        if _is_skippable(nd):
            j += 1
            continue
        if isinstance(nd, Assign) and isinstance(nd.lhs, Name):
            f = _split_field(nd.lhs.name)
            if f is not None and f[0] == lhs_parent:
                pairs.append((f[1], nd.rhs))
                j += 1
                continue
        break

    if len(pairs) < 2:
        return None, i

    suffixes = [p[0] for p in pairs]
    if not _valid_suffix_sequence(suffixes):
        return None, i

    rhs_exprs = [p[1] for p in pairs]

    # Try: all RHS are Name("rhs_parent.same_suffix") or aliased Regs — field copy
    rhs_parent = None
    valid_field = True
    for suf, rhs in zip(suffixes, rhs_exprs):
        # Accept Name("parent.suffix") or Regs(..., alias="parent.suffix")
        field_name = None
        if isinstance(rhs, Name):
            field_name = rhs.name
        elif isinstance(rhs, Regs) and rhs.alias:
            field_name = rhs.alias
        if field_name is not None:
            f = _split_field(field_name)
            if f is not None and f[1] == suf:
                if rhs_parent is None:
                    rhs_parent = f[0]
                elif f[0] != rhs_parent:
                    valid_field = False
                    break
            else:
                valid_field = False
                break
        else:
            valid_field = False
            break

    if valid_field and rhs_parent is not None:
        dbg("typesimp", f"  mb_assign: {lhs_parent} = {rhs_parent}")
        result = Assign(n0.ea, Name(lhs_parent), Name(rhs_parent))
        result.source_nodes = nodes[i:j]
        return result, j

    # Try: all RHS are integer constants (Const or Name with numeric literal)
    const_vals = [_rhs_as_const(rhs) for rhs in rhs_exprs]
    if all(v is not None for v in const_vals):
        value = 0
        for cv in const_vals:
            value = (value << 8) | (cv & 0xFF)
        dbg("typesimp", f"  mb_assign: {lhs_parent} = {hex(value)} (const)")
        result = Assign(n0.ea, Name(lhs_parent), Const(value))
        result.source_nodes = nodes[i:j]
        return result, j

    # Try: 16-bit RMW add-with-carry from 8051 MUL result.
    # Pattern (lo first or hi first):
    #   var.lo = var.lo + Cast('uint8_t', full_expr)
    #   var.hi = var.hi + mul_hi_expr [+ C]
    # where mul_hi_expr is typically B (MUL high byte) and full_expr is the
    # complete 16-bit MUL expression (e.g. _var2 * 3).
    # → var = var + full_expr  (CompoundAssign var += full_expr)
    #
    # Only handles the 2-byte case.
    if set(suffixes) == {"hi", "lo"} and len(rhs_exprs) == 2:
        if suffixes == ["lo", "hi"]:
            lo_rhs, hi_rhs = rhs_exprs
        else:
            hi_rhs, lo_rhs = rhs_exprs
        full_expr = _extract_mul_addc_pair(lhs_parent, hi_rhs, lo_rhs)
        if full_expr is not None:
            dbg("typesimp",
                f"  mb_assign: {lhs_parent} += {full_expr.render()} (mul-addc)")
            node = CompoundAssign(n0.ea, Name(lhs_parent), "+=", full_expr)
            node.source_nodes = nodes[i:j]
            return node, j

        # Try: general 16-bit add-with-carry
        # var.lo = var.lo + lo_addend; var.hi = var.hi + hi_addend + C
        if suffixes == ["lo", "hi"]:
            lo_rhs2, hi_rhs2 = rhs_exprs
        else:
            hi_rhs2, lo_rhs2 = rhs_exprs
        combined = _extract_add_carry_pair(lhs_parent, lo_rhs2, hi_rhs2)
        if combined is not None:
            dbg("typesimp",
                f"  mb_assign: {lhs_parent} += {combined.render()} (add-carry)")
            node = CompoundAssign(n0.ea, Name(lhs_parent), "+=", combined)
            node.source_nodes = nodes[i:j]
            return node, j

    return None, i


def _try_collapse_pair_store(nodes: List[HIRNode], i: int):
    """
    Recognise the pattern where a multi-register result is stored back to a
    named hi/lo byte-field variable pair:

        Rhi:Rlo = expr          (multi-reg Assign with exactly 2 registers)
        var.hi  = Rhi           (store high byte)
        var.lo  = Rlo           (store low byte)

    and optionally an immediately following compound-assign into the same var:

        var += addend

    Collapses to:
        var = expr              (or  var = expr + addend  when += follows)

    The two byte-field stores may appear in either hi-then-lo or lo-then-hi
    order.  DPTR++ and similar skippable nodes between the three main nodes
    are silently dropped, as in _try_collapse.
    """
    n0 = nodes[i]
    if not (isinstance(n0, Assign)
            and isinstance(n0.lhs, Regs)
            and not n0.lhs.is_single
            and len(n0.lhs.names) == 2):
        return None, i

    rhi, rlo = n0.lhs.names   # first name = high byte, second = low byte
    rhs_expr = n0.rhs

    hi_found = lo_found = False
    var_name = None
    j = i + 1

    while j < len(nodes) and not (hi_found and lo_found):
        nd = nodes[j]
        if _is_skippable(nd):
            j += 1
            continue
        if isinstance(nd, Assign) and isinstance(nd.lhs, Name):
            f = _split_field(nd.lhs.name)
            if f is not None:
                parent, suf = f
                if var_name is not None and parent != var_name:
                    break
                rhs = nd.rhs
                if suf == 'hi' and not hi_found:
                    if isinstance(rhs, Regs) and rhs.is_single and rhs.name == rhi:
                        var_name = parent
                        hi_found = True
                        j += 1
                        continue
                elif suf == 'lo' and not lo_found:
                    if isinstance(rhs, Regs) and rhs.is_single and rhs.name == rlo:
                        var_name = parent
                        lo_found = True
                        j += 1
                        continue
        break

    if not (hi_found and lo_found and var_name is not None):
        return None, i

    end_j = j  # index past the last consumed byte-field store

    # Optionally consume an immediately following  var += addend
    k = j
    while k < len(nodes) and _is_skippable(nodes[k]):
        k += 1
    if k < len(nodes):
        nxt = nodes[k]
        if (isinstance(nxt, CompoundAssign)
                and isinstance(nxt.lhs, Name)
                and nxt.lhs.name == var_name
                and nxt.op == '+='):
            rhs_expr = BinOp(rhs_expr, '+', nxt.rhs)
            end_j = k + 1

    dbg("typesimp", f"  mb_assign: {var_name} = ... (pair-store)")
    result = Assign(n0.ea, Name(var_name), rhs_expr)
    result.source_nodes = nodes[i:end_j]
    return result, end_j


def _extract_mul_addc_pair(var_name: str, hi_rhs, lo_rhs) -> Optional['Expr']:
    """
    Detect the 8051 16-bit add-with-carry RMW pattern from a MUL result:

      var.hi = var.hi + <hi_expr> [+ C]
      var.lo = var.lo + Cast('uint8_t', full_expr)

    Returns full_expr if the pattern matches, else None.

    Rules:
      - lo_rhs must be BinOp('+', Name(var.lo)_or_alias, Cast('uint8_t', full_expr))
        or BinOp('+', Cast('uint8_t', full_expr), Name(var.lo)_or_alias)
      - hi_rhs must be BinOp('+', Name(var.hi)_or_alias, hi_addend)
        where hi_addend contains Reg('B') (the MUL high-byte register) with
        optional Reg('C') carry
      - The cast-stripped full_expr in lo_rhs matches the value that B would
        hold (same expression tree, or B is the only non-C non-var addend in hi)
    """
    # Unwrap lo: var.lo + Cast('uint8_t', full_expr)
    full_expr = _unwrap_rmw_add(f"{var_name}.lo", lo_rhs, _is_uint8_cast)
    if full_expr is None:
        return None

    # Unwrap hi: var.hi + <hi_addend [+ C]>
    hi_addend = _unwrap_rmw_add(f"{var_name}.hi", hi_rhs, None)
    if hi_addend is None:
        return None

    # hi_addend must be B, or B+C, or C+B (C is the carry from the lo add)
    if not _is_b_plus_carry(hi_addend):
        return None

    return full_expr


def _name_matches(expr, field_name: str) -> bool:
    """True if expr is Name(field_name) or Regs with alias==field_name."""
    if isinstance(expr, Name) and expr.name == field_name:
        return True
    if isinstance(expr, Regs) and expr.alias == field_name:
        return True
    return False


def _unwrap_rmw_add(field_name: str, rhs, inner_pred) -> Optional['Expr']:
    """
    Match  rhs == field_name + inner  (or inner + field_name).
    If inner_pred is given, apply it to inner and return its result.
    If inner_pred is None, return inner directly.
    Returns None on mismatch.
    """
    if not isinstance(rhs, BinOp) or rhs.op != '+':
        return None
    if _name_matches(rhs.lhs, field_name):
        inner = rhs.rhs
    elif _name_matches(rhs.rhs, field_name):
        inner = rhs.lhs
    else:
        return None
    if inner_pred is not None:
        return inner_pred(inner)
    return inner


def _is_uint8_cast(expr) -> Optional['Expr']:
    """If expr is Cast('uint8_t', inner), return inner; else None."""
    if isinstance(expr, Cast) and expr.type_str == 'uint8_t':
        return expr.inner
    return None


def _is_b_plus_carry(expr) -> bool:
    """True if expr is B, B+C, or C+B (8051 ADDC high-byte pattern)."""
    if isinstance(expr, Regs) and expr.is_single and expr.name == 'B':
        return True
    if isinstance(expr, BinOp) and expr.op == '+':
        lhs_b = isinstance(expr.lhs, Regs) and expr.lhs.is_single and expr.lhs.name == 'B'
        rhs_b = isinstance(expr.rhs, Regs) and expr.rhs.is_single and expr.rhs.name == 'B'
        lhs_c = isinstance(expr.lhs, Regs) and expr.lhs.is_single and expr.lhs.name == 'C'
        rhs_c = isinstance(expr.rhs, Regs) and expr.rhs.is_single and expr.rhs.name == 'C'
        return (lhs_b and rhs_c) or (lhs_c and rhs_b)
    return False


def _has_carry(expr) -> bool:
    """True if Reg('C') appears anywhere in expr."""
    if isinstance(expr, Regs) and expr.is_single and expr.name == 'C':
        return True
    if isinstance(expr, BinOp):
        return _has_carry(expr.lhs) or _has_carry(expr.rhs)
    return False


def _extract_add_carry_pair(var_name: str, lo_rhs, hi_rhs) -> Optional['Expr']:
    """
    Recognise the 16-bit add-with-carry RMW pattern:

      var.lo = var.lo + lo_addend                (no carry)
      var.hi = (var.hi + hi_addend) + C          (carry in outermost position)
             OR var.hi + (hi_addend + C)

    Returns a combined addend expression, or None if the pattern doesn't match.

    Combined addend rules:
      - Name(x.lo) + Name(x.hi) → Name(x)
      - Const(lo) + Const(hi)   → Const((hi << 8) | lo)
      - Otherwise               → None (cannot safely combine)
    """
    lo_field = f"{var_name}.lo"
    hi_field = f"{var_name}.hi"

    # --- Lo side: var.lo + lo_addend (or lo_addend + var.lo), no carry ---
    if not isinstance(lo_rhs, BinOp) or lo_rhs.op != '+':
        return None
    if _name_matches(lo_rhs.lhs, lo_field):
        lo_addend = lo_rhs.rhs
    elif _name_matches(lo_rhs.rhs, lo_field):
        lo_addend = lo_rhs.lhs
    else:
        return None
    if _has_carry(lo_addend):
        return None

    # --- Hi side: (var.hi + hi_addend) + C  OR  var.hi + (hi_addend + C) ---
    if not isinstance(hi_rhs, BinOp) or hi_rhs.op != '+':
        return None

    hi_addend: Optional['Expr'] = None
    # Case A: outermost `+` has Reg('C') on the right
    if hi_rhs.rhs == Reg('C'):
        inner = hi_rhs.lhs
        if isinstance(inner, BinOp) and inner.op == '+':
            if _name_matches(inner.lhs, hi_field):
                hi_addend = inner.rhs
            elif _name_matches(inner.rhs, hi_field):
                hi_addend = inner.lhs
    # Case B: var.hi is the outermost left operand, right contains carry
    elif _name_matches(hi_rhs.lhs, hi_field):
        tail = hi_rhs.rhs
        if isinstance(tail, BinOp) and tail.op == '+' and tail.rhs == Reg('C'):
            hi_addend = tail.lhs
        elif isinstance(tail, BinOp) and tail.op == '+' and tail.lhs == Reg('C'):
            hi_addend = tail.rhs
    # Case C: var.hi is the outermost right operand
    elif _name_matches(hi_rhs.rhs, hi_field):
        tail = hi_rhs.lhs
        if isinstance(tail, BinOp) and tail.op == '+' and tail.rhs == Reg('C'):
            hi_addend = tail.lhs
        elif isinstance(tail, BinOp) and tail.op == '+' and tail.lhs == Reg('C'):
            hi_addend = tail.rhs

    if hi_addend is None or _has_carry(hi_addend):
        return None

    # --- Combine lo_addend and hi_addend ---
    # Name(x.lo) + Name(x.hi) → Name(x)
    if isinstance(lo_addend, Name) and isinstance(hi_addend, Name):
        f_lo = _split_field(lo_addend.name)
        f_hi = _split_field(hi_addend.name)
        if (f_lo and f_hi
                and f_lo[0] == f_hi[0]
                and f_lo[1] == 'lo'
                and f_hi[1] == 'hi'):
            return Name(f_lo[0])

    # Const(lo_val) + Const(hi_val) → Const combined
    if isinstance(lo_addend, Const) and isinstance(hi_addend, Const):
        return Const((hi_addend.value << 8) | (lo_addend.value & 0xFF))

    return None


def _parse_addc_terms(expr, suffix):
    """Walk a '+'-only BinOp tree extracting named byte-field operands.

    Returns (names, carries, extras) where:
      names   = ordered list of parent names for Name(parent.suffix) nodes
      carries = count of Reg('C') / Regs(('C',)) terms
      extras  = list of Expr nodes that are neither Name(*.suffix) nor carry
    """
    names = []
    carries = 0
    extras = []

    def _walk(e):
        nonlocal carries
        if isinstance(e, BinOp) and e.op == '+':
            _walk(e.lhs)
            _walk(e.rhs)
        elif isinstance(e, Regs) and e.is_single and e.name == 'C':
            carries += 1
        elif isinstance(e, Name):
            f = _split_field(e.name)
            if f is not None and f[1] == suffix:
                names.append(f[0])
            else:
                extras.append(e)
        else:
            extras.append(e)

    _walk(expr)
    return names, carries, extras


def _try_fix_stale_reg_pair_addc(nodes: List[HIRNode], i: int, out: List[HIRNode]):
    """
    Recognise the 16-bit add-with-carry pattern on register LHS and fix any
    stale TypedAssign for the same register pair in the already-emitted output.

    Pattern:
      nodes[i]:   Assign(Regs((R_lo,)), a.lo + b.lo [+ byte_extras...])
      nodes[i+j]: Assign(Regs((R_hi,)), a.hi + b.hi + C [+ C per extra lo addend])

    If a TypedAssign(Regs((R_hi, R_lo)), stale_expr) exists in out[], it is
    updated in-place with the correct combined expression  a + b [+ byte_extras],
    and both the R_lo and R_hi assignment nodes are consumed (not appended to out).

    Returns (True, new_i) on success, or (False, i) if the pattern doesn't match
    or no stale TypedAssign exists for the pair.
    """
    n0 = nodes[i]
    if not (isinstance(n0, Assign)
            and not isinstance(n0, TypedAssign)
            and isinstance(n0.lhs, Regs)
            and n0.lhs.is_single):
        return False, i

    r_lo = n0.lhs.name
    lo_names, lo_carries, lo_extras = _parse_addc_terms(n0.rhs, 'lo')
    if not lo_names or lo_carries != 0:
        return False, i

    # Find next non-skippable node (hi-byte assign)
    j = i + 1
    while j < len(nodes) and _is_skippable(nodes[j]):
        j += 1
    if j >= len(nodes):
        return False, i

    n1 = nodes[j]
    if not (isinstance(n1, Assign)
            and not isinstance(n1, TypedAssign)
            and isinstance(n1.lhs, Regs)
            and n1.lhs.is_single):
        return False, i

    r_hi = n1.lhs.name
    hi_names, hi_carries, hi_extras = _parse_addc_terms(n1.rhs, 'hi')
    if not hi_names or hi_extras:
        return False, i
    if sorted(lo_names) != sorted(hi_names):
        return False, i
    # Each extra lo-byte addend contributes one carry to hi
    if hi_carries != 1 + len(lo_extras):
        return False, i

    # Build the combined 16-bit expression, preserving lo operand order
    combined = Name(lo_names[0])
    for name in lo_names[1:]:
        combined = BinOp(combined, '+', Name(name))
    for extra in lo_extras:
        combined = BinOp(combined, '+', extra)

    # Look backward in out[] for a TypedAssign whose LHS is Regs({r_hi, r_lo})
    pair_set = {r_hi, r_lo}
    for k in range(len(out) - 1, -1, -1):
        nd = out[k]
        if (isinstance(nd, TypedAssign)
                and isinstance(nd.lhs, Regs)
                and not nd.lhs.is_single
                and set(nd.lhs.names) == pair_set):
            new_ta = TypedAssign(nd.ea, nd.type_str, nd.lhs, combined)
            nd.copy_meta_to(new_ta)
            new_ta.source_nodes = [n0, n1] + list(nd.source_nodes or [nd])
            out[k] = new_ta
            alias = nd.lhs.alias or f"{r_hi}:{r_lo}"
            dbg("typesimp",
                f"  mb_assign: fixed stale {alias!r} TypedAssign "
                f"→ {combined.render()!r}")
            return True, j + 1

    return False, i


def collapse_mb_assigns(nodes: List[HIRNode]) -> List[HIRNode]:
    """
    Second-pass collapse of byte-field assignment sequences.

    Recurses into all structured node bodies (IfNode, WhileNode, ForNode,
    DoWhileNode, SwitchNode) via map_bodies so nested byte-field sequences
    inside any control structure are also collapsed.
    """
    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]

        # Recurse into structured nodes via map_bodies (returns new node for
        # structured types, self for leaf types).
        mapped = node.map_bodies(collapse_mb_assigns)
        if mapped is not node:
            out.append(mapped)
            i += 1
            continue

        collapsed, new_i = _try_collapse_rmw_inc(nodes, i)
        if collapsed is None:
            collapsed, new_i = _try_collapse_pair_store(nodes, i)
        if collapsed is None:
            collapsed, new_i = _try_collapse(nodes, i)
        if collapsed is not None:
            out.append(collapsed)
            i = new_i
        else:
            fixed, new_i = _try_fix_stale_reg_pair_addc(nodes, i, out)
            if fixed:
                i = new_i
            else:
                out.append(node)
                i += 1

    return out
