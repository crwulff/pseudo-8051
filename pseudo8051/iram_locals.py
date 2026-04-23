"""
iram_locals.py — Per-function IRAM local variable storage (IDA netnode backend).

Declarations persist in the IDA database across sessions without modifying
any source file.  One netnode per function, keyed by the function's start EA.

Usage (via the pseudo8051 top-level re-exports):

    import pseudo8051
    pseudo8051.set_iram_local(here(), 0x34, 'ptr', 'uint8_t')
    pseudo8051.list_iram_locals(here())
    pseudo8051.del_iram_local(here(), 0x34)

'here()' can be any address inside the target function.
"""

from dataclasses import dataclass
from typing import List

from pseudo8051.constants import dbg


@dataclass
class IRAMLocalVar:
    """A typed IRAM local variable scoped to a specific function."""
    name: str   # variable name shown in pseudocode, e.g. 'ptr'
    type: str   # C99 type, e.g. 'uint8_t'
    addr: int   # IRAM address, e.g. 0x34


# ── IDA netnode storage ───────────────────────────────────────────────────────
# One netnode per function: "$ pseudo8051.iram_locals.0x{func_ea:08x}"
# Each entry:  supset(iram_addr, "name\ttype", 'I')

_NN_PREFIX = "$ pseudo8051.iram_locals."
_NN_TAG    = 'I'


def _iram_locals_nn(func_ea: int, create: bool = False):
    """
    Open (and optionally create) the netnode storing IRAM locals for func_ea.
    Returns None when create=False and no node exists, to avoid polluting the
    database with empty netnodes.
    """
    try:
        import ida_netnode
        name = f"{_NN_PREFIX}{func_ea:#010x}"
        nn = ida_netnode.netnode()
        is_new = nn.create(name)
        if is_new and not create:
            nn.kill()
            return None
        return nn
    except Exception as e:
        dbg("iram_locals", f"_iram_locals_nn({func_ea:#x}): {e}")
        return None


def get_iram_locals(func_ea: int) -> List[IRAMLocalVar]:
    """Return declared IRAM local variables for the function at func_ea."""
    try:
        import idc
        nn = _iram_locals_nn(func_ea, create=False)
        if nn is None:
            return []
        result: List[IRAMLocalVar] = []
        alt = nn.supfirst(_NN_TAG)
        while alt != idc.BADADDR:
            val = nn.supstr(alt, _NN_TAG)
            if val:
                parts = val.split('\t', 1)
                if len(parts) == 2:
                    result.append(IRAMLocalVar(name=parts[0], type=parts[1],
                                               addr=int(alt)))
            alt = nn.supnext(alt, _NN_TAG)
        return result
    except Exception as e:
        dbg("iram_locals", f"get_iram_locals({func_ea:#x}): {e}")
        return []


def set_iram_local(func_ea_or_here: int, iram_addr: int,
                   name: str, type_str: str) -> None:
    """
    Declare (or update) a typed IRAM local variable for a function.

    func_ea_or_here may be the function's start address or any address inside
    the function — the function entry point is resolved automatically.
    """
    try:
        import ida_funcs
        fn = ida_funcs.get_func(func_ea_or_here)
        if fn is None:
            print(f"pseudo8051: {func_ea_or_here:#x} is not inside a function")
            return
        func_ea = fn.start_ea
        nn = _iram_locals_nn(func_ea, create=True)
        nn.supset(iram_addr, f"{name}\t{type_str}", _NN_TAG)
        fname = ida_funcs.get_func_name(func_ea) or hex(func_ea)
        print(f"pseudo8051: {type_str} {name} @ IRAM[{iram_addr:#04x}]"
              f" declared in {fname}")
    except Exception as e:
        print(f"pseudo8051: set_iram_local error: {e}")


def del_iram_local(func_ea_or_here: int, iram_addr: int) -> None:
    """Remove a declared IRAM local variable."""
    try:
        import ida_funcs
        fn = ida_funcs.get_func(func_ea_or_here)
        func_ea = fn.start_ea if fn else func_ea_or_here
        nn = _iram_locals_nn(func_ea, create=False)
        if nn is not None:
            nn.supdel(iram_addr, _NN_TAG)
    except Exception as e:
        print(f"pseudo8051: del_iram_local error: {e}")


def list_iram_locals(func_ea_or_here: int) -> None:
    """Print declared IRAM local variables for the function."""
    try:
        import ida_funcs
        fn = ida_funcs.get_func(func_ea_or_here)
        func_ea = fn.start_ea if fn else func_ea_or_here
        fname   = ida_funcs.get_func_name(func_ea) or hex(func_ea)
    except Exception:
        func_ea = func_ea_or_here
        fname   = hex(func_ea)

    locs = get_iram_locals(func_ea)
    if not locs:
        print(f"pseudo8051: no IRAM locals declared in {fname}")
        return
    print(f"pseudo8051: IRAM locals in {fname}:")
    for lv in locs:
        print(f"  {lv.type:12s} {lv.name}  @ IRAM[{lv.addr:#04x}]")
