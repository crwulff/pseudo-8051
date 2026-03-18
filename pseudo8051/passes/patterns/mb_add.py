"""
passes/patterns/mb_add.py — MultiByteAddPattern.

Recognises the 8051 idiom for adding a 16-bit constant to a register pair
and storing the result byte-by-byte into a declared XRAM local:

    A = Rlo;               load low source byte
    A += lo_const;         8-bit add  (ADD A, #lo)
    Rlo = A;               save result
    A = Rhi;               load high source byte
    A += hi_const + C;     8-bit add-with-carry  (ADDC A, #hi)
    XRAM[sym_hi] = A;      write high byte to XRAM local
    DPTR++;
    [A = Rlo;]             optional reload of saved low result
    XRAM[sym_lo] = A;      write low byte to XRAM local

    →   var = <src_pair_expr> + combined_const;

where combined_const = (hi_const << 8) | lo_const, and src_pair_expr is the
typed C expression for Rhi:Rlo (e.g. ``(uint16_t)dividend`` when R6:R7 are
the low word of a 32-bit parameter).

When combined_const is zero the addition is omitted:
    var = <src_pair_expr>;
"""

import re
from typing import Dict, List, Optional

from pseudo8051.ir.hir import HIRNode, Statement
from pseudo8051.constants import dbg
from pseudo8051.passes.patterns.base   import Pattern, Match, Simplify
from pseudo8051.passes.patterns._utils import (
    VarInfo, _type_bytes, _const_str, _replace_pairs,
)

_RE_A_FROM_REG = re.compile(r'^A = (\w+);$')
_RE_ADD_CONST  = re.compile(r'^A \+= (0x[0-9a-fA-F]+|\d+);$')
_RE_ADDC_CONST = re.compile(r'^A \+= (0x[0-9a-fA-F]+|\d+) \+ C;$')
_RE_REG_FROM_A = re.compile(r'^(\w+) = A;$')
_RE_XRAM_WRITE = re.compile(r'^XRAM\[(.+?)\] = A;$')
_RE_DPTR_INC   = re.compile(r'^DPTR\+\+;$')
_RE_DPTR_SET   = re.compile(r'^DPTR = .+?;')


def _parse_const(s: str) -> int:
    return int(s, 16) if s.lower().startswith('0x') else int(s)


def _src_expr(Rhi: str, Rlo: str,
              reg_map: Dict[str, VarInfo],
              target_type: str) -> str:
    """
    Build a typed C expression for the Rhi:Rlo register pair.

    If both registers belong to the same register-based VarInfo and are
    consecutive positions within its regs tuple, emit a cast of the parent
    variable (e.g. ``(uint16_t)dividend``).  Otherwise synthesise the pair
    name and apply register-pair substitution so named variables appear if
    the pair is itself in reg_map.
    """
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
            if lo_pos == hi_pos + 1:     # Rhi:Rlo are consecutive bytes
                src_bytes = _type_bytes(vinfo.type)
                dst_bytes = _type_bytes(target_type)
                if src_bytes == dst_bytes:
                    return vinfo.name
                return f"({target_type}){vinfo.name}"

    # Fallback: pair name — _replace_pairs resolves it if it is in reg_map
    pair = Rhi + Rlo
    if pair in reg_map and not reg_map[pair].xram_sym:
        return _replace_pairs(pair, reg_map)
    return pair


def _is_lo_companion(sym_hi: str, sym_lo: str,
                     hi_vinfo: VarInfo,
                     reg_map: Dict[str, VarInfo]) -> bool:
    """
    Return True if sym_lo is the byte immediately after sym_hi in the same
    declared XRAM local (identified via the per-byte VarInfo entries).
    """
    lo_binfo = reg_map.get(f"_byte_{sym_lo}")
    if lo_binfo is None or not lo_binfo.is_byte_field:
        return False
    return lo_binfo.name.startswith(hi_vinfo.name + ".")


class MultiByteAddPattern(Pattern):
    """
    Collapse 8051 16-bit ADD+ADDC arithmetic + XRAM byte writes into a
    single typed assignment to a declared XRAM local.
    """

    def match(self,
              nodes:    List[HIRNode],
              i:        int,
              reg_map:  Dict[str, VarInfo],
              simplify: Simplify) -> Optional[Match]:
        n = len(nodes)

        def stext(j: int) -> str:
            return nodes[j].text if j < n and isinstance(nodes[j], Statement) else ""

        j = i

        # 1. A = Rlo;
        m = _RE_A_FROM_REG.match(stext(j))
        if not m:
            return None
        Rlo = m.group(1)
        j += 1
        dbg("mb-add", f"[{i}] step1 ok  Rlo={Rlo!r}  next={stext(j)!r}")

        # 2. A += lo_const;  (ADD A, #lo)
        m = _RE_ADD_CONST.match(stext(j))
        if not m:
            dbg("mb-add", f"[{i}] step2 FAIL  expected ADD_CONST, got {stext(j)!r}")
            return None
        lo_const = _parse_const(m.group(1))
        j += 1
        dbg("mb-add", f"[{i}] step2 ok  lo_const={lo_const:#x}")

        # 3. Rlo = A;  (save low-byte result so it can be reloaded later)
        m = _RE_REG_FROM_A.match(stext(j))
        if not m or m.group(1) != Rlo:
            dbg("mb-add", f"[{i}] step3 FAIL  expected {Rlo}=A, got {stext(j)!r}")
            return None
        j += 1

        # 4. A = Rhi;
        m = _RE_A_FROM_REG.match(stext(j))
        if not m or m.group(1) == Rlo:
            dbg("mb-add", f"[{i}] step4 FAIL  expected A=Rhi, got {stext(j)!r}")
            return None
        Rhi = m.group(1)
        j += 1
        dbg("mb-add", f"[{i}] step4 ok  Rhi={Rhi!r}")

        # 5. A += hi_const + C;  (ADDC A, #hi)
        m = _RE_ADDC_CONST.match(stext(j))
        if not m:
            dbg("mb-add", f"[{i}] step5 FAIL  expected ADDC_CONST, got {stext(j)!r}")
            return None
        hi_const = _parse_const(m.group(1))
        j += 1
        dbg("mb-add", f"[{i}] step5 ok  hi_const={hi_const:#x}  next={stext(j)!r}")

        # 5b. Optional: DPTR = sym;  (MOV DPL/DPH setup lifted as DPTR = addr;)
        if _RE_DPTR_SET.match(stext(j)):
            dbg("mb-add", f"[{i}] step5b skip DPTR setup {stext(j)!r}")
            j += 1

        # 6. XRAM[sym_hi] = A;
        m = _RE_XRAM_WRITE.match(stext(j))
        if not m:
            dbg("mb-add", f"[{i}] step6 FAIL  expected XRAM_WRITE, got {stext(j)!r}")
            return None
        sym_hi = m.group(1).strip()
        j += 1
        dbg("mb-add", f"[{i}] step6 ok  sym_hi={sym_hi!r}")

        # sym_hi must be a declared full-var XRAM local of at least 2 bytes
        hi_vinfo = reg_map.get(sym_hi)
        if hi_vinfo is None:
            dbg("mb-add", f"[{i}] step6 FAIL  sym_hi {sym_hi!r} not in reg_map  keys={[k for k in reg_map if 'EXT' in k or 'byte' in k]}")
            return None
        if not hi_vinfo.xram_sym or hi_vinfo.is_byte_field or _type_bytes(hi_vinfo.type) < 2:
            dbg("mb-add", f"[{i}] step6 FAIL  vinfo check: xram_sym={hi_vinfo.xram_sym!r} is_byte_field={hi_vinfo.is_byte_field} type={hi_vinfo.type!r}")
            return None

        # 7. DPTR++;
        if not _RE_DPTR_INC.match(stext(j)):
            dbg("mb-add", f"[{i}] step7 FAIL  expected DPTR++, got {stext(j)!r}")
            return None
        j += 1

        # 8. Optional reload: A = Rlo;
        m = _RE_A_FROM_REG.match(stext(j))
        if m and m.group(1) == Rlo:
            j += 1

        # 9. XRAM[sym_lo] = A;
        m = _RE_XRAM_WRITE.match(stext(j))
        if not m:
            dbg("mb-add", f"[{i}] step9 FAIL  expected XRAM_WRITE(lo), got {stext(j)!r}")
            return None
        sym_lo = m.group(1).strip()
        j += 1

        # Verify sym_lo is the immediate low-byte companion of sym_hi
        if not _is_lo_companion(sym_hi, sym_lo, hi_vinfo, reg_map):
            dbg("mb-add", f"[{i}] step9 FAIL  lo companion check: sym_lo={sym_lo!r} hi_name={hi_vinfo.name!r}")
            return None

        # Build replacement
        src  = _src_expr(Rhi, Rlo, reg_map, hi_vinfo.type)
        cval = (hi_const << 8) | lo_const

        if cval == 0:
            text = f"{hi_vinfo.name} = {src};"
        else:
            cstr = _const_str(cval, hi_vinfo.type)
            text = f"{hi_vinfo.name} = {src} + {cstr};"

        dbg("typesimp", f"  mb-add: {text}  (nodes {i}–{j-1}, ea={nodes[i].ea:#x})")
        return ([Statement(nodes[i].ea, text)], j)
