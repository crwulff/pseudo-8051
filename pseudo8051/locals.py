"""
locals.py — Per-function XRAM local variable storage (IDA netnode backend).

Declarations persist in the IDA database across sessions without modifying
any source file.  One netnode per function, keyed by the function's start EA.

Usage (via the pseudo8051 top-level re-exports):

    import pseudo8051
    pseudo8051.set_local(here(), 0xdc8a, 'count', 'int16_t')
    pseudo8051.list_locals(here())
    pseudo8051.del_local(here(), 0xdc8a)

'here()' can be any address inside the target function.
"""

from dataclasses import dataclass
from typing import List

from pseudo8051.constants import dbg


@dataclass
class LocalVar:
    """A typed XRAM local variable scoped to a specific function."""
    name: str   # variable name shown in pseudocode, e.g. 'count'
    type: str   # C99 type, e.g. 'int16_t'
    addr: int   # XRAM base address, e.g. 0xdc8a


# ── IDA netnode storage ───────────────────────────────────────────────────────
# One netnode per function: "$ pseudo8051.locals.0x{func_ea:08x}"
# Each entry:  supset(xram_addr, "name\ttype", 'L')

_NN_PREFIX = "$ pseudo8051.locals."
_NN_TAG    = 'L'


def _locals_nn(func_ea: int, create: bool = False):
    """
    Open (and optionally create) the netnode storing locals for func_ea.

    Uses nn.create(name) rather than the netnode(name, alt, bool) constructor
    because the 3-argument constructor form is unreliable across IDA versions.
    When create=False and the node doesn't exist, returns None to avoid
    polluting the database with empty netnodes.
    """
    try:
        import ida_netnode
        name = f"{_NN_PREFIX}{func_ea:#010x}"
        nn = ida_netnode.netnode()
        is_new = nn.create(name)   # True = just created (empty); False = existed
        if is_new and not create:
            nn.kill()              # drop the empty node we just made
            return None
        return nn
    except Exception as e:
        dbg("locals", f"_locals_nn({func_ea:#x}): {e}")
        return None


def get_locals(func_ea: int) -> List[LocalVar]:
    """
    Return declared XRAM local variables for the function at func_ea.
    Returns an empty list if none have been declared or on any error.
    """
    try:
        import idc
        nn = _locals_nn(func_ea, create=False)
        if nn is None:
            return []
        result: List[LocalVar] = []
        alt = nn.supfirst(_NN_TAG)
        while alt != idc.BADADDR:
            val = nn.supstr(alt, _NN_TAG)
            if val:
                parts = val.split('\t', 1)
                if len(parts) == 2:
                    result.append(LocalVar(name=parts[0], type=parts[1],
                                           addr=int(alt)))
            alt = nn.supnext(alt, _NN_TAG)
        return result
    except Exception as e:
        dbg("locals", f"get_locals({func_ea:#x}): {e}")
        return []


def set_local(func_ea_or_here: int, xram_addr: int,
              name: str, type_str: str) -> None:
    """
    Declare (or update) a typed XRAM local variable for a function.

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
        nn = _locals_nn(func_ea, create=True)
        nn.supset(xram_addr, f"{name}\t{type_str}", _NN_TAG)
        fname = ida_funcs.get_func_name(func_ea) or hex(func_ea)
        print(f"pseudo8051: {type_str} {name} @ {xram_addr:#06x}"
              f" declared in {fname}")
    except Exception as e:
        print(f"pseudo8051: set_local error: {e}")


def del_local(func_ea_or_here: int, xram_addr: int) -> None:
    """Remove a declared XRAM local variable."""
    try:
        import ida_funcs
        fn = ida_funcs.get_func(func_ea_or_here)
        func_ea = fn.start_ea if fn else func_ea_or_here
        nn = _locals_nn(func_ea, create=False)
        if nn is not None:
            nn.supdel(xram_addr, _NN_TAG)
    except Exception as e:
        print(f"pseudo8051: del_local error: {e}")


def list_locals(func_ea_or_here: int) -> None:
    """Print declared XRAM local variables for the function."""
    try:
        import ida_funcs
        fn = ida_funcs.get_func(func_ea_or_here)
        func_ea = fn.start_ea if fn else func_ea_or_here
        fname   = ida_funcs.get_func_name(func_ea) or hex(func_ea)
    except Exception:
        func_ea = func_ea_or_here
        fname   = hex(func_ea)

    locs = get_locals(func_ea)
    if not locs:
        print(f"pseudo8051: no locals declared in {fname}")
        return
    print(f"pseudo8051: locals in {fname}:")
    for lv in locs:
        print(f"  {lv.type:12s} {lv.name}  @ {lv.addr:#06x}")
