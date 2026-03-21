"""
passes/patterns/mb_add.py — MultiByteAddPattern.

Recognises the 8051 idiom for adding a 16-bit constant to a register pair
and storing the result byte-by-byte into a declared XRAM local.
"""

from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement, Assign, CompoundAssign, ExprStmt
from pseudo8051.ir.expr import Reg, Const, XRAMRef, UnaryOp
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _type_bytes, _const_str, _replace_pairs,
)


def _parse_const(s: str) -> int:
    return int(s, 16) if s.lower().startswith('0x') else int(s)


def _node_is_a_from_reg(node: HIRNode) -> Optional[str]:
    """If node is 'A = Rn;', return Rn; else None."""
    if isinstance(node, Assign):
        if node.lhs == Reg("A") and isinstance(node.rhs, Reg):
            return node.rhs.name
    return None


def _node_add_const(node: HIRNode) -> Optional[int]:
    """If node is 'A += const;', return const; else None."""
    if isinstance(node, CompoundAssign):
        if node.lhs == Reg("A") and node.op == "+=" and isinstance(node.rhs, Const):
            return node.rhs.value
    return None


def _node_addc_const(node: HIRNode) -> Optional[int]:
    """If node is 'A += const + C;', return const; else None."""
    if isinstance(node, CompoundAssign):
        if node.lhs == Reg("A") and node.op == "+=":
            from pseudo8051.ir.expr import BinOp
            rhs = node.rhs
            if isinstance(rhs, BinOp) and rhs.op == "+" and rhs.rhs == Reg("C"):
                if isinstance(rhs.lhs, Const):
                    return rhs.lhs.value
    return None


def _node_is_reg_from_a(node: HIRNode, expected_reg: str) -> bool:
    """True if node is 'expected_reg = A;'."""
    if isinstance(node, Assign):
        return node.lhs == Reg(expected_reg) and node.rhs == Reg("A")
    return False


def _node_xram_write_a(node: HIRNode) -> Optional[str]:
    """If node is 'XRAM[sym] = A;', return sym; else None."""
    if isinstance(node, Assign):
        if isinstance(node.lhs, XRAMRef) and node.rhs == Reg("A"):
            return node.lhs.inner.render()
    return None


def _node_is_dptr_inc(node: HIRNode) -> bool:
    if isinstance(node, ExprStmt):
        if isinstance(node.expr, UnaryOp):
            return node.expr.op == "++" and node.expr.operand == Reg("DPTR")
    return False


def _node_is_dptr_set(node: HIRNode) -> bool:
    if isinstance(node, Assign):
        return node.lhs == Reg("DPTR")
    return False


def _src_expr(Rhi: str, Rlo: str,
              reg_map: Dict[str, VarInfo],
              target_type: str) -> str:
    rhi_info = reg_map.get(Rhi)
    rlo_info = reg_map.get(Rlo)

    if (rhi_info is not None and rlo_info is not None
            and rhi_info is rlo_info
            and not rhi_info.xram_sym):
        vinfo = rhi_info
        regs  = list(vinfo.regs)
        try:
            hi_pos = regs.index(Rhi)
            lo_pos = regs.index(Rlo)
        except ValueError:
            pass
        else:
            if lo_pos == hi_pos + 1:
                src_bytes = _type_bytes(vinfo.type)
                dst_bytes = _type_bytes(target_type)
                if src_bytes == dst_bytes:
                    return vinfo.name
                return f"({target_type}){vinfo.name}"

    pair = Rhi + Rlo
    if pair in reg_map and not reg_map[pair].xram_sym:
        return _replace_pairs(pair, reg_map)
    return pair


def _is_lo_companion(sym_lo: str,
                     hi_vinfo: VarInfo,
                     reg_map: Dict[str, VarInfo]) -> bool:
    lo_binfo = reg_map.get(f"_byte_{sym_lo}")
    if lo_binfo is None or not lo_binfo.is_byte_field:
        return False
    return lo_binfo.name.startswith(hi_vinfo.name + ".")


class MultiByteAddPattern(Pattern):
    """Collapse 8051 16-bit ADD+ADDC arithmetic + XRAM byte writes into a typed assignment."""

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        n_total = len(nodes)
        j = i

        def get(k): return nodes[k] if k < n_total else None

        # 1. A = Rlo;
        Rlo = _node_is_a_from_reg(get(j)) if get(j) else None
        if not Rlo: return None
        j += 1
        dbg("mb-add", f"[{i}] step1 ok  Rlo={Rlo!r}")

        # 2. A += lo_const;
        lo_const = _node_add_const(get(j)) if get(j) else None
        if lo_const is None: return None
        j += 1
        dbg("mb-add", f"[{i}] step2 ok  lo_const={lo_const:#x}")

        # 3. Rlo = A;
        if not get(j) or not _node_is_reg_from_a(get(j), Rlo): return None
        j += 1

        # 4. A = Rhi;
        node4 = get(j)
        Rhi = _node_is_a_from_reg(node4) if node4 else None
        if not Rhi or Rhi == Rlo: return None
        j += 1
        dbg("mb-add", f"[{i}] step4 ok  Rhi={Rhi!r}")

        # 5. A += hi_const + C;
        hi_const = _node_addc_const(get(j)) if get(j) else None
        if hi_const is None: return None
        j += 1

        # 5b. Optional DPTR setup
        if get(j) and _node_is_dptr_set(get(j)):
            j += 1

        # 6. XRAM[sym_hi] = A;
        sym_hi = _node_xram_write_a(get(j)) if get(j) else None
        if not sym_hi: return None
        j += 1

        hi_vinfo = reg_map.get(sym_hi)
        if hi_vinfo is None or not hi_vinfo.xram_sym or hi_vinfo.is_byte_field or _type_bytes(hi_vinfo.type) < 2:
            return None

        # 7. DPTR++;
        if not get(j) or not _node_is_dptr_inc(get(j)): return None
        j += 1

        # 8. Optional reload: A = Rlo;
        node8 = get(j)
        if node8 and _node_is_a_from_reg(node8) == Rlo:
            j += 1

        # 9. XRAM[sym_lo] = A;
        sym_lo = _node_xram_write_a(get(j)) if get(j) else None
        if not sym_lo: return None
        j += 1

        if not _is_lo_companion(sym_lo, hi_vinfo, reg_map):
            return None

        src  = _src_expr(Rhi, Rlo, reg_map, hi_vinfo.type)
        cval = (hi_const << 8) | lo_const

        if cval == 0:
            text = f"{hi_vinfo.name} = {src};"
        else:
            cstr = _const_str(cval, hi_vinfo.type)
            text = f"{hi_vinfo.name} = {src} + {cstr};"

        dbg("typesimp", f"  mb-add: {text}  (nodes {i}–{j-1}, ea={nodes[i].ea:#x})")
        return ([Statement(nodes[i].ea, text)], j)
