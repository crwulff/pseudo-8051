"""
prototypes.py — Function prototype database for the 8051 pseudocode generator.

Prototypes are read from IDA's own type system first (set via the 'y' key in
IDA Pro).  The manual PROTOTYPES dict below is consulted as an override — use
it when a function uses a non-standard register convention or you want to
rename parameters.

IDA's internal type names (e.g. '__int16', 'unsigned __int16') are normalised
to C99 names (uint16_t etc.) so the return-register table can match them.

Return-register inference from C type (standard 8051 convention):
    void        →  ()                    no return value
    bool        →  ('C',)               carry flag
    uint8_t     →  ('A',)               accumulator
    uint16_t    →  ('R6', 'R7')         word pair
    uint32_t    →  ('R4','R5','R6','R7') dword quad

To override for a specific function (e.g. returns uint16_t in R0:R1):

    PROTOTYPES['code_1_Foo'] = FuncProto(
        return_type = 'uint16_t',
        return_regs = ('R0', 'R1'),
        params      = [Param('count', 'uint8_t', ('A',))],
    )
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from pseudo8051.constants import dbg, PARAM_REGS


@dataclass
class Param:
    """One parameter to a function."""
    name: str                   # name shown in pseudocode, e.g. 'value'
    type: str                   # C type, e.g. 'uint16_t'
    regs: Tuple[str, ...] = ()  # physical registers (informational; may be empty)


@dataclass
class FuncProto:
    """Prototype (signature) for one function."""
    return_type: str                        # 'void', 'bool', 'uint8_t', 'uint16_t', …
    return_regs: Tuple[str, ...] = ()       # registers holding return value
    params:      List[Param]    = field(default_factory=list)


def return_expr(proto: "FuncProto") -> str:
    """
    C expression for the return value — concatenation of return_regs names:
    ('R6','R7') → 'R6R7',  ('C',) → 'C'.
    """
    return "".join(proto.return_regs)


# ── Standard 8051 calling-convention register allocation ─────────────────────

_PROTO_REG_POOL = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]

_PROTO_TYPE_BYTES: dict = {
    "void": 0, "bool": 1,
    "int8_t": 1, "uint8_t": 1,
    "int16_t": 2, "uint16_t": 2,
    "int32_t": 4, "uint32_t": 4,
}


def param_regs(proto: "FuncProto") -> List[Tuple[str, ...]]:
    """
    Return the physical register tuple for each parameter using the standard
    8051 calling convention (R7 downward).  Explicit Param.regs take priority.
    """
    pool = list(_PROTO_REG_POOL)
    result: List[Tuple[str, ...]] = []
    for p in proto.params:
        if p.regs:
            result.append(p.regs)
            pool = [r for r in pool if r not in p.regs]
        else:
            size = _PROTO_TYPE_BYTES.get(p.type, 0)
            if size == 0 or size > len(pool):
                result.append(())
                continue
            regs = tuple(pool[-size:])
            pool = pool[:-size]
            result.append(regs)
    return result


# ── IDA → C99 type name normalisation ────────────────────────────────────────

_IDA_TYPE_MAP = {
    "void":              "void",
    "bool":              "bool",
    "_Bool":             "bool",
    "_BOOL":             "bool",
    "char":              "uint8_t",
    "unsigned char":     "uint8_t",
    "signed char":       "int8_t",
    "__int8":            "int8_t",
    "unsigned __int8":   "uint8_t",
    "short":             "int16_t",
    "unsigned short":    "uint16_t",
    "__int16":           "int16_t",
    "unsigned __int16":  "uint16_t",
    "int":               "int32_t",
    "unsigned int":      "uint32_t",
    "long":              "int32_t",
    "unsigned long":     "uint32_t",
    "__int32":           "int32_t",
    "unsigned __int32":  "uint32_t",
    # already-normalised names pass through unchanged
    "int8_t":   "int8_t",   "uint8_t":   "uint8_t",
    "int16_t":  "int16_t",  "uint16_t":  "uint16_t",
    "int32_t":  "int32_t",  "uint32_t":  "uint32_t",
}


def _norm(ida_type: str) -> str:
    """Normalise an IDA type string to a C99 name, or return it unchanged."""
    s = ida_type.strip()
    return _IDA_TYPE_MAP.get(s, s)


# ── Standard 8051 return-register table (keyed by normalised C99 names) ──────

_RETURN_REGS: dict = {
    "void":     (),
    "bool":     ("C",),
    "int8_t":   ("A",),
    "uint8_t":  ("A",),
    "int16_t":  ("R6", "R7"),
    "uint16_t": ("R6", "R7"),
    "int32_t":  ("R4", "R5", "R6", "R7"),
    "uint32_t": ("R4", "R5", "R6", "R7"),
}


# ── Parse idc.get_type() string as fallback ───────────────────────────────────
# idc.get_type() returns e.g. "__int16 __cdecl func(__int16 a1, __int8 a2)"
# __usercall format: "rettype __usercall func@<Rn>(type name@<Rn>, ...)"

# Matches: "type name" or "type name@<REG>"
_RE_PARAM = re.compile(r'(.+?)\s+(\w+)(?:@<([^>]+)>)?\s*$')
# idc.get_type() format: "rettype(params)" or "rettype __cdecl(params)"
# __usercall return reg: "rettype@<Rn> __usercall func(params)"
# — no function name, calling convention may appear before the '('
_RE_GETTYPE = re.compile(
    r'^(.+?)'                                                                    # return type (may include @<Rn>)
    r'(?:\s+(?:__cdecl|__stdcall|__fastcall|__far|__usercall|__userpurge))?'    # optional calling conv
    r'\s*\(([^)]*)\)\s*;?\s*$'                                                  # ( params )
)
# Capture @<Rn> from a return type string, e.g. "__int16@<R6R7>"
_RE_RET_LOC = re.compile(r'@<([^>]+)>')


_REG_RE = re.compile(r'(R[0-7]|DPTR|[AC])')


def _regs_from_loc_str(loc_str: str) -> Tuple[str, ...]:
    """
    Parse a register location string from IDA's __usercall annotation.
    Handles single registers ('R6'), pairs ('R6R7'), and quads ('R4R5R6R7').
    Returns a tuple of register names, or () if not recognised.
    """
    parts: List[str] = []
    for m in _REG_RE.finditer(loc_str.strip()):
        reg = m.group(1)
        if reg in PARAM_REGS or reg in ("A", "C", "DPTR"):
            parts.append(reg)
    return tuple(parts) if parts else ()


def _regs_from_argloc(aloc, type_size: int) -> Tuple[str, ...]:
    """
    Extract register name(s) from an IDA argloc_t / retloc_t.
    For ALOC_REG1 the register number encodes a base register; consecutive
    registers are inferred from the type size (1→1 reg, 2→2, 4→4).
    For ALOC_REG2 (explicit pair like R2:R1 in __usercall), both register
    numbers are read directly.
    Returns () if the location is not a register location.
    """
    try:
        import ida_typeinf as _idt
        import ida_idp
        atype = aloc.atype()
        regnames = ida_idp.ph.regnames

        if atype == _idt.ALOC_REG1:
            rn = aloc.reg1()
            if 0 <= rn < len(regnames):
                base = regnames[rn].upper()
                # For multi-byte types, expand consecutive registers
                if type_size > 1:
                    bm = re.match(r'^R(\d)$', base)
                    if bm:
                        idx = int(bm.group(1))
                        regs = tuple(f"R{idx + k}" for k in range(type_size)
                                     if idx + k <= 7)
                        if len(regs) == type_size:
                            return regs
                return _regs_from_loc_str(base)

        # ALOC_REG2 — explicit register pair, e.g. R2:R1 in __usercall
        #
        # IDA stores reg1() as the lower-numbered register (e.g. R1 for R2:R1).
        # The prototype notation "A:B" follows the DX:AX convention where A is
        # more significant (hi byte), and A is the higher-numbered register.
        # So we must reverse the [reg1, reg2] order to get [hi, lo].
        aloc_reg2 = getattr(_idt, 'ALOC_REG2', None)
        if aloc_reg2 is not None and atype == aloc_reg2:
            rn1 = aloc.reg1()
            # Try multiple accessor names for the second register
            rn2 = -1
            for attr in ('reg2', 'regoff'):
                try:
                    rn2 = int(getattr(aloc, attr)())
                    break
                except Exception:
                    pass
            parts = []
            for rn in [rn1, rn2]:
                if 0 <= rn < len(regnames):
                    name = regnames[rn].upper()
                    if name in PARAM_REGS or name in ("A", "C", "DPTR"):
                        parts.append(name)
            if parts:
                return tuple(reversed(parts))  # higher-numbered reg first = hi byte
    except Exception:
        pass
    return ()


def _parse_type_string(type_str: str, func_name: str) -> Optional["FuncProto"]:
    """
    Parse a type string from idc.get_type() — format 'rettype(params)' —
    into a FuncProto.  Returns None if parsing fails.
    Handles __usercall with @<Rn> register annotations on return type and params.
    """
    m = _RE_GETTYPE.match(type_str.strip())
    if not m:
        dbg("proto", f"{func_name}: can't parse type string {type_str!r}")
        return None

    ret_raw    = m.group(1).strip()
    params_raw = m.group(2).strip()

    # Extract @<Rn> from return type if present (e.g. "__int16@<R6R7>")
    ret_loc_m = _RE_RET_LOC.search(ret_raw)
    ret_loc_str = ret_loc_m.group(1) if ret_loc_m else ""
    ret_type_str = _RE_RET_LOC.sub("", ret_raw).strip()
    ret_type = _norm(ret_type_str)

    if ret_loc_str:
        ret_regs = _regs_from_loc_str(ret_loc_str)
        dbg("proto", f"{func_name}: usercall return loc {ret_loc_str!r} → {ret_regs}")
    else:
        ret_regs = _RETURN_REGS.get(ret_type, ())

    params: List[Param] = []
    if params_raw and params_raw.lower() != "void":
        for i, part in enumerate(params_raw.split(",")):
            part = part.strip()
            pm = _RE_PARAM.match(part)
            if pm:
                ptype = _norm(pm.group(1).strip())
                pname = pm.group(2).strip()
                ploc  = pm.group(3) or ""
            else:
                ptype = _norm(part)
                pname = f"arg{i}"
                ploc  = ""
            pregs = _regs_from_loc_str(ploc) if ploc else ()
            params.append(Param(name=pname, type=ptype, regs=pregs))

    return FuncProto(return_type=ret_type, return_regs=ret_regs, params=params)


# ── Read from IDA type system ('y' key) ──────────────────────────────────────

def _proto_from_ida(name: str) -> Optional["FuncProto"]:
    """
    Try to build a FuncProto from IDA's type info for the named function.
    Tries the structured tinfo API first, then falls back to idc.get_type().
    Returns None if IDA has no type info for it.
    """
    dbg("proto", f"{name}: querying IDA type info …")

    try:
        import idc
        import ida_name
        import ida_nalt
        import ida_typeinf  # noqa: F401 — keep for ImportError early exit
    except ImportError as e:
        dbg("proto", f"{name}: IDA modules not available — {e}")
        return None

    ea = ida_name.get_name_ea(idc.BADADDR, name)
    dbg("proto", f"{name}: ea = {hex(ea) if ea != idc.BADADDR else 'BADADDR'}")
    if ea == idc.BADADDR:
        return None

    # ── Try structured tinfo API ──────────────────────────────────────────
    proto = None
    try:
        import ida_typeinf as _idt
        tif = _idt.tinfo_t()
        got = ida_nalt.get_tinfo(tif, ea)
        dbg("proto", f"{name}: get_tinfo={got}  is_func={tif.is_func() if got else 'n/a'}")
        if got and tif.is_func():
            fi = _idt.func_type_data_t()
            gfd = tif.get_func_details(fi)
            dbg("proto", f"{name}: get_func_details={gfd}")
            if gfd:
                raw_ret  = str(fi.rettype)
                ret_type = _norm(raw_ret)
                # Check for explicit return register location (usercall convention)
                ret_size = _PROTO_TYPE_BYTES.get(ret_type, 0)
                try:
                    ret_regs_uc = _regs_from_argloc(fi.retloc, ret_size)
                    if ret_regs_uc:
                        ret_regs = ret_regs_uc
                        dbg("proto", f"{name}: usercall retloc → {ret_regs}")
                    else:
                        ret_regs = _RETURN_REGS.get(ret_type, ())
                except Exception:
                    ret_regs = _RETURN_REGS.get(ret_type, ())
                params: List[Param] = []
                for i in range(fi.size()):
                    arg   = fi[i]
                    pname = arg.name if arg.name else f"arg{i}"
                    ptype = _norm(str(arg.type))
                    # Extract register from argloc if __usercall
                    arg_size = _PROTO_TYPE_BYTES.get(ptype, 0)
                    try:
                        pregs = _regs_from_argloc(arg.argloc, arg_size)
                        if pregs:
                            dbg("proto", f"{name}: arg{i} argloc → {pregs}")
                    except Exception:
                        pregs = ()
                    params.append(Param(name=pname, type=ptype, regs=pregs))
                proto = FuncProto(return_type=ret_type,
                                  return_regs=ret_regs, params=params)
                dbg("proto", f"{name}: tinfo → ret={ret_type!r} ({raw_ret!r})  "
                             f"ret_regs={ret_regs}  "
                             f"params=[{', '.join(f'{p.type} {p.name}' for p in params)}]")
    except Exception as e:
        dbg("proto", f"{name}: tinfo API error — {e}")

    # ── Supplement missing regs from idc.get_type() string ───────────────
    # If the tinfo path built a proto but some params have regs=() (e.g. because
    # ALOC_REG2 isn't decoded), try to fill them in from the type string.
    if proto and any(not p.regs for p in proto.params):
        try:
            type_str = idc.get_type(ea)
            if type_str:
                str_proto = _parse_type_string(type_str, name)
                if str_proto and len(str_proto.params) == len(proto.params):
                    new_params = list(proto.params)
                    for i, (tp, sp) in enumerate(zip(proto.params, str_proto.params)):
                        if not tp.regs and sp.regs:
                            new_params[i] = Param(tp.name, tp.type, sp.regs)
                            dbg("proto", f"{name}: arg{i} regs supplemented "
                                        f"from string: {sp.regs}")
                    proto = FuncProto(proto.return_type, proto.return_regs,
                                      new_params)
        except Exception as e:
            dbg("proto", f"{name}: supplement error — {e}")

    if proto:
        return proto

    # ── Fallback: idc.get_type() string ──────────────────────────────────
    try:
        type_str = idc.get_type(ea)
        dbg("proto", f"{name}: idc.get_type → {type_str!r}")
        if type_str:
            proto = _parse_type_string(type_str, name)
            if proto:
                dbg("proto", f"{name}: parsed → ret={proto.return_type!r}  "
                             f"ret_regs={proto.return_regs}  "
                             f"params=[{', '.join(f'{p.type} {p.name}' for p in proto.params)}]")
    except Exception as e:
        dbg("proto", f"{name}: idc.get_type error — {e}")

    return proto


# ── Manual override table ─────────────────────────────────────────────────────
# Entries here take priority over IDA type info.

PROTOTYPES: dict = {}


# ── Enum registry ─────────────────────────────────────────────────────────────
# Maps enum type name → {integer_value: "symbolic_name"}.
# Used by SwitchCaseAnnotator to annotate case labels.
# IDA's own enum definitions are also consulted automatically as a fallback.
#
# Example:
#   ENUMS["PtrType"] = {0xFE: "PtrType_RAM_FE", 0xFF: "PtrType_Code",
#                       0x00: "PtrType_RAM_00", 0x01: "PtrType_XRAM"}

ENUMS: dict = {}


def get_enum_name(type_str: str, value: int) -> Optional[str]:
    """
    Return the symbolic name for *value* in enum *type_str*, or None.

    Checks ENUMS first (manual overrides), then falls back to IDA's enum API.
    """
    manual = ENUMS.get(type_str)
    if manual is not None:
        return manual.get(value)
    # IDA 9.0 enum fallback (ida_enum was removed; use idc)
    try:
        import idc
        eid = idc.get_enum(type_str)
        if eid == idc.BADADDR:
            return None
        # BADADDR used as bmask selects non-bitmask (regular) enum members
        cid = idc.get_enum_member(eid, value, 0, idc.BADADDR)
        if cid != idc.BADADDR:
            return idc.get_enum_member_name(cid)
    except Exception as exc:
        dbg("proto", f"get_enum_name({type_str!r}, {hex(value)}): {exc}")
    return None


def get_proto(name: str) -> Optional[FuncProto]:
    """
    Look up a function prototype by IDA name.

    Priority:
      1. Manual PROTOTYPES entry (override / non-standard convention)
      2. IDA type system (set via 'y' in IDA Pro)
      3. None — unknown function
    """
    manual = PROTOTYPES.get(name)
    if manual is not None:
        dbg("proto", f"{name}: using manual PROTOTYPES entry")
        return manual
    return _proto_from_ida(name)
