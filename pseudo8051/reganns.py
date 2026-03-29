"""
reganns.py — Per-function register annotation storage (IDA netnode backend).

Declarations persist in the IDA database across sessions without modifying
any source file.  One netnode per function, keyed by the function's start EA.

Usage (via the pseudo8051 top-level re-exports):

    import pseudo8051
    pseudo8051.set_regann(here(), here(), 'R6,R7', 'count', 'uint16_t')
    pseudo8051.list_reganns(here())
    pseudo8051.del_regann(here(), here(), 'R6,R7')

'here()' can be any address inside the target function.
The second here() is the address of the instruction to annotate.
"""

from dataclasses import dataclass
from typing import List, Tuple

from pseudo8051.constants import dbg


@dataclass
class RegAnn:
    """A typed register annotation at a specific instruction address."""
    ea:   int                  # instruction address
    regs: Tuple[str, ...]      # ('R6',) or ('R6', 'R7')
    name: str                  # variable name shown in pseudocode
    type: str                  # C99 type, e.g. 'uint16_t'


# ── IDA netnode storage ───────────────────────────────────────────────────────
# One netnode per function: "$ pseudo8051.reganns.0x{func_ea:08x}"
# Each entry: supset(insn_ea, "R6,R7\tcount\tuint16_t\nR4\tflags\tuint8_t", 'A')
# Multiple annotations at the same EA are separated by newlines.

_NN_PREFIX = "$ pseudo8051.reganns."
_NN_TAG    = 'A'


def _reganns_nn(func_ea: int, create: bool = False):
    """
    Open (and optionally create) the netnode storing reg annotations for func_ea.

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
        dbg("reganns", f"_reganns_nn({func_ea:#x}): {e}")
        return None


def _parse_entry(insn_ea: int, raw: str) -> List[RegAnn]:
    """Parse a newline-separated multi-line entry into RegAnn objects."""
    result = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t', 2)
        if len(parts) != 3:
            continue
        regs_str, name, type_str = parts
        regs = tuple(r.strip() for r in regs_str.split(',') if r.strip())
        if regs and name and type_str:
            result.append(RegAnn(ea=insn_ea, regs=regs, name=name, type=type_str))
    return result


def _regs_key(regs_str: str) -> str:
    """Normalize a regs string to canonical form (sorted, stripped)."""
    return ",".join(r.strip() for r in regs_str.split(',') if r.strip())


def get_reganns(func_ea: int) -> List[RegAnn]:
    """
    Return declared register annotations for the function at func_ea.
    Returns an empty list if none have been declared or on any error.
    """
    try:
        import idc
        nn = _reganns_nn(func_ea, create=False)
        if nn is None:
            return []
        result: List[RegAnn] = []
        alt = nn.supfirst(_NN_TAG)
        while alt != idc.BADADDR:
            raw = nn.supstr(alt, _NN_TAG)
            if raw:
                result.extend(_parse_entry(int(alt), raw))
            alt = nn.supnext(alt, _NN_TAG)
        return result
    except Exception as e:
        dbg("reganns", f"get_reganns({func_ea:#x}): {e}")
        return []


def set_regann(func_ea_or_here: int, insn_ea: int,
               regs_str: str, name: str, type_str: str) -> None:
    """
    Declare (or update) a register annotation at insn_ea for a function.

    func_ea_or_here may be the function's start address or any address inside
    the function — the function entry point is resolved automatically.
    regs_str is comma-separated: "R6" or "R6,R7".
    """
    try:
        import ida_funcs
        fn = ida_funcs.get_func(func_ea_or_here)
        if fn is None:
            print(f"pseudo8051: {func_ea_or_here:#x} is not inside a function")
            return
        func_ea = fn.start_ea
        nn = _reganns_nn(func_ea, create=True)

        # Read existing entry at insn_ea
        raw = nn.supstr(insn_ea, _NN_TAG) or ""
        canon_new = _regs_key(regs_str)

        # Rebuild lines, replacing matching regs line or appending
        lines = [l for l in raw.splitlines() if l.strip()]
        replaced = False
        for i, line in enumerate(lines):
            parts = line.split('\t', 2)
            if len(parts) == 3:
                canon_existing = _regs_key(parts[0])
                if canon_existing == canon_new:
                    lines[i] = f"{canon_new}\t{name}\t{type_str}"
                    replaced = True
                    break
        if not replaced:
            lines.append(f"{canon_new}\t{name}\t{type_str}")

        nn.supset(insn_ea, "\n".join(lines), _NN_TAG)
        fname = ida_funcs.get_func_name(func_ea) or hex(func_ea)
        print(f"pseudo8051: {type_str} {name} in {canon_new} @ {insn_ea:#x}"
              f" declared in {fname}")
    except Exception as e:
        print(f"pseudo8051: set_regann error: {e}")


def del_regann(func_ea_or_here: int, insn_ea: int, regs_str: str) -> None:
    """Remove a declared register annotation."""
    try:
        import ida_funcs
        fn = ida_funcs.get_func(func_ea_or_here)
        func_ea = fn.start_ea if fn else func_ea_or_here
        nn = _reganns_nn(func_ea, create=False)
        if nn is None:
            return
        raw = nn.supstr(insn_ea, _NN_TAG) or ""
        canon_del = _regs_key(regs_str)
        lines = [l for l in raw.splitlines() if l.strip()]
        new_lines = []
        for line in lines:
            parts = line.split('\t', 2)
            if len(parts) == 3 and _regs_key(parts[0]) == canon_del:
                continue
            new_lines.append(line)
        if new_lines:
            nn.supset(insn_ea, "\n".join(new_lines), _NN_TAG)
        else:
            nn.supdel(insn_ea, _NN_TAG)
    except Exception as e:
        print(f"pseudo8051: del_regann error: {e}")


def list_reganns(func_ea_or_here: int) -> None:
    """Print declared register annotations for the function."""
    try:
        import ida_funcs
        fn = ida_funcs.get_func(func_ea_or_here)
        func_ea = fn.start_ea if fn else func_ea_or_here
        fname   = ida_funcs.get_func_name(func_ea) or hex(func_ea)
    except Exception:
        func_ea = func_ea_or_here
        fname   = hex(func_ea)

    anns = get_reganns(func_ea)
    if not anns:
        print(f"pseudo8051: no register annotations declared in {fname}")
        return
    print(f"pseudo8051: register annotations in {fname}:")
    for ra in anns:
        regs = ",".join(ra.regs)
        print(f"  {ra.type:12s} {ra.name} in {regs} @ {ra.ea:#010x}")
