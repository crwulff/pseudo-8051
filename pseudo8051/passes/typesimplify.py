"""
passes/typesimplify.py — TypeAwareSimplifier: fold typed multi-byte patterns.

Uses the function prototype to:
  1. Build a register → variable-name map (explicit regs or inferred from the
     standard 8051 calling convention).
  2. Replace register-pair occurrences (e.g. R6R7) in Statement text with the
     parameter or return-variable name.
  3. Recognise and collapse common multi-byte patterns:
       sign-bit test:    A = R_hi;  if (ACC.7) { … }  →  if (var < 0) { … }
       16-bit negation:  7-statement SUBB sequence     →  var = -var;

Patterns are matched at any nesting level (inside IfNode/WhileNode bodies too).
Individual register references (R6, R7) are intentionally left as-is unless
consumed by a named pattern, to avoid spurious replacements.
"""

import re
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from pseudo8051.ir.hir    import HIRNode, Statement, IfNode, WhileNode, ForNode
from pseudo8051.passes    import OptimizationPass
from pseudo8051.constants import dbg

if TYPE_CHECKING:
    from pseudo8051.ir.function import Function
    from pseudo8051.prototypes  import FuncProto, Param


# ── Type helpers ──────────────────────────────────────────────────────────────

def _type_bytes(t: str) -> int:
    if t in ("bool", "uint8_t",  "int8_t",  "char"):  return 1
    if t in ("uint16_t", "int16_t"):                   return 2
    if t in ("uint32_t", "int32_t"):                   return 4
    return 0


def _is_signed(t: str) -> bool:
    return t in ("int8_t", "int16_t", "int32_t")


# ── Standard 8051 calling-convention register assignment ─────────────────────
# Allocate consecutive registers from R7 downward, largest block first.

_REG_POOL = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]


def _assign_regs(params) -> List[Tuple[str, ...]]:
    """
    Assign physical registers to a parameter list using the standard 8051
    high-register convention.  Returns a parallel list of register tuples.
    """
    pool = list(_REG_POOL)
    result = []
    for p in params:
        size = _type_bytes(p.type)
        if size == 0 or size > len(pool):
            result.append(())
            continue
        regs = tuple(pool[-size:])   # take top `size` registers
        pool = pool[:-size]
        result.append(regs)
    return result


# ── VarInfo: one named variable covering one or more registers ────────────────

class VarInfo:
    def __init__(self, name: str, type_str: str, regs: Tuple[str, ...]):
        self.name      = name
        self.type      = type_str
        self.regs      = regs           # high → low order, e.g. ('R6', 'R7')
        self.pair_name = "".join(regs)  # e.g. 'R6R7'

    @property
    def hi(self) -> Optional[str]:
        """Highest register (most-significant byte), or None for single-byte."""
        return self.regs[0] if len(self.regs) >= 2 else None

    @property
    def lo(self) -> Optional[str]:
        """Lowest register (least-significant byte)."""
        return self.regs[-1] if self.regs else None


def _build_reg_map(proto: "FuncProto") -> Dict[str, VarInfo]:
    """
    Build a dict mapping register names → VarInfo for all prototype params
    and the return registers.

    Only multi-register keys (e.g. 'R6R7') are added for text substitution;
    individual register keys are still present so pattern matchers can look
    them up.
    """
    params   = proto.params
    assigned: List[Tuple[str, ...]] = []
    for p in params:
        assigned.append(p.regs if p.regs else ())

    # Fill missing assignments from convention
    needs = [i for i, r in enumerate(assigned) if not r]
    if needs:
        inferred = _assign_regs(params)
        for i in needs:
            assigned[i] = inferred[i]

    reg_map: Dict[str, VarInfo] = {}

    for p, regs in zip(params, assigned):
        if not regs:
            continue
        info = VarInfo(p.name, p.type, regs)
        for r in regs:
            reg_map[r] = info
        if len(regs) > 1:
            reg_map[info.pair_name] = info

    # Return registers — map to the overlapping param if any, else 'retval'
    if proto.return_regs:
        ret_info = None
        for r in proto.return_regs:
            if r in reg_map:
                ret_info = reg_map[r]
                break
        if ret_info is None:
            ret_info = VarInfo("retval", proto.return_type, proto.return_regs)
        pair = "".join(proto.return_regs)
        if pair not in reg_map:
            reg_map[pair] = ret_info
        for r in proto.return_regs:
            if r not in reg_map:
                reg_map[r] = ret_info

    return reg_map


# ── Text-level token substitution ─────────────────────────────────────────────

def _replace_pairs(text: str, reg_map: Dict[str, VarInfo]) -> str:
    """
    Replace register-pair tokens (keys with len > 2, e.g. R6R7) with the
    variable name.  Tries longest keys first; uses word-boundary matching.
    Individual registers (R6, R7, A, C …) are left untouched here.
    """
    for key in sorted((k for k in reg_map if len(k) > 2), key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(key) + r"\b", reg_map[key].name, text)
    return text


# ── Pattern matchers ──────────────────────────────────────────────────────────

def _match_sign_load(node: HIRNode, reg_map: Dict[str, VarInfo]) -> Optional[VarInfo]:
    """
    Match 'A = R_hi;' where R_hi is the high byte of a signed ≥16-bit param.
    Returns VarInfo if matched, else None.
    """
    if not isinstance(node, Statement):
        return None
    m = re.match(r"^A = (\w+);$", node.text)
    if not m:
        return None
    info = reg_map.get(m.group(1))
    if (info and info.hi == m.group(1)
            and _type_bytes(info.type) >= 2 and _is_signed(info.type)):
        return info
    return None


# Pattern for 16-bit two's-complement negation:
#   C = 0;
#   A = 0;
#   A -= R_lo + C;  /* borrow */
#   R_lo = A;
#   A = 0;
#   A -= R_hi + C;  /* borrow */
#   R_hi = A;
_RE_SUBB = re.compile(r"^A -= (\w+) \+ C;")
_RE_STOR = re.compile(r"^(\w+) = A;$")


def _match_neg16(nodes: List[HIRNode], reg_map: Dict[str, VarInfo],
                 start: int) -> Optional[Tuple[VarInfo, int]]:
    """
    Try to match the 7-statement 16-bit negation at nodes[start:].
    Returns (VarInfo, 7) on success, else None.
    """
    if start + 7 > len(nodes):
        return None
    ns = nodes[start:start + 7]

    # Positions 0, 1, 4 are fixed literals
    fixed = {0: r"^C = 0;$", 1: r"^A = 0;$", 4: r"^A = 0;$"}
    for idx, pat in fixed.items():
        if not (isinstance(ns[idx], Statement) and re.match(pat, ns[idx].text)):
            return None

    m2 = isinstance(ns[2], Statement) and _RE_SUBB.match(ns[2].text)
    m3 = isinstance(ns[3], Statement) and _RE_STOR.match(ns[3].text)
    m5 = isinstance(ns[5], Statement) and _RE_SUBB.match(ns[5].text)
    m6 = isinstance(ns[6], Statement) and _RE_STOR.match(ns[6].text)
    if not (m2 and m3 and m5 and m6):
        return None

    r_lo = m2.group(1)
    r_hi = m5.group(1)
    if m3.group(1) != r_lo or m6.group(1) != r_hi:
        return None

    info_lo = reg_map.get(r_lo)
    info_hi = reg_map.get(r_hi)
    if (info_lo and info_hi and info_lo is info_hi
            and info_lo.lo == r_lo and info_lo.hi == r_hi):
        return (info_lo, 7)
    return None


# ── Constant-load-into-register-group ────────────────────────────────────────
# Matches sequences like:
#   A = 0;  R7 = 0x60;  R6 = 0x6d;  R5 = A;  R4 = A;
# and collapses them to a single constant assignment.

_RE_ASSIGN_IMM = re.compile(r"^(\w+) = (0x[0-9a-fA-F]+|\d+);$")
_RE_ASSIGN_REG = re.compile(r"^(\w+) = (\w+);$")


def _parse_int(s: str) -> int:
    return int(s, 16) if s.lower().startswith("0x") else int(s)


def _const_str(value: int, type_str: str) -> str:
    size = _type_bytes(type_str)
    if size >= 4: return f"0x{value:08x}"
    if size == 2: return f"0x{value:04x}"
    return hex(value) if value > 9 else str(value)


def _match_const_group(nodes: List[HIRNode], start: int,
                       vinfo: VarInfo) -> Optional[Tuple[int, int]]:
    """
    Starting at `start`, scan for a sequence of statements that together
    assign a compile-time constant to every register of vinfo.

    Handles:
      Rn = literal;          — direct constant load
      A  = literal;          — A used as a zero/constant carrier
      Rn = A;                — A value forwarded to register

    Returns (combined_value, end_index) on success, else None.
    """
    if len(vinfo.regs) < 2:
        return None

    regs_needed = set(vinfo.regs)
    reg_values: Dict[str, int] = {}
    a_value: Optional[int] = None
    i = start
    # Allow up to 2× the register count + 2 extra steps for A loads
    max_i = min(len(nodes), start + len(regs_needed) * 2 + 2)

    while i < max_i and len(reg_values) < len(regs_needed):
        node = nodes[i]
        if not isinstance(node, Statement):
            break
        text = node.text

        m_imm = _RE_ASSIGN_IMM.match(text)
        if m_imm:
            dst, val_s = m_imm.group(1), m_imm.group(2)
            val = _parse_int(val_s)
            if dst == "A":
                a_value = val
                i += 1
                continue
            if dst in regs_needed and dst not in reg_values:
                reg_values[dst] = val
                i += 1
                continue
            break   # constant load to an unrelated register — stop

        m_reg = _RE_ASSIGN_REG.match(text)
        if m_reg:
            dst, src = m_reg.group(1), m_reg.group(2)
            if dst in regs_needed and src == "A" and a_value is not None \
                    and dst not in reg_values:
                reg_values[dst] = a_value
                i += 1
                continue
            if dst == "A":
                a_value = None  # A overwritten with unknown value
            break   # unrecognised register move — stop

        break   # any other statement — stop

    if regs_needed != set(reg_values.keys()):
        return None

    # Combine byte values (vinfo.regs is high→low, e.g. R4 R5 R6 R7)
    value = 0
    for reg in vinfo.regs:
        value = (value << 8) | (reg_values[reg] & 0xFF)
    return (value, i)


# ── Recursive simplifier ──────────────────────────────────────────────────────

def _simplify(nodes: List[HIRNode], reg_map: Dict[str, VarInfo]) -> List[HIRNode]:
    """
    Walk `nodes`, applying pattern transforms and pair renaming.
    Returns the transformed list.
    """
    out: List[HIRNode] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]

        # ── Sign-bit test: A = R_hi; if (ACC.7) {…} ──────────────────────
        sign_info = _match_sign_load(node, reg_map)
        if sign_info and i + 1 < len(nodes):
            nxt = nodes[i + 1]
            if isinstance(nxt, IfNode) and nxt.condition == "ACC.7":
                dbg("typesimp", f"  sign-test: {sign_info.name} < 0")
                out.append(IfNode(
                    ea         = nxt.ea,
                    condition  = f"{sign_info.name} < 0",
                    then_nodes = _simplify(nxt.then_nodes, reg_map),
                    else_nodes = _simplify(nxt.else_nodes, reg_map),
                ))
                i += 2
                continue

        # ── Constant load into register group ────────────────────────────
        # Try largest groups first (4-byte before 2-byte) to avoid partial match
        const_matched = False
        for vinfo in sorted({v for v in reg_map.values() if len(v.regs) >= 2},
                             key=lambda v: len(v.regs), reverse=True):
            cm = _match_const_group(nodes, i, vinfo)
            if cm is None:
                continue
            value, end_i = cm
            const_s = _const_str(value, vinfo.type)
            dbg("typesimp", f"  const-load: {vinfo.name} = {const_s}")
            # Fold into return if immediately followed by 'return <var>;'.
            # The lookahead node may still carry the pre-substitution pair name
            # (e.g. 'return R4R5R6R7;') so check both forms.
            ret_texts = {f"return {vinfo.name};", f"return {vinfo.pair_name};"}
            if (end_i < len(nodes)
                    and isinstance(nodes[end_i], Statement)
                    and nodes[end_i].text in ret_texts):
                out.append(Statement(node.ea, f"return {const_s};"))
                i = end_i + 1
            else:
                # Declare with type so 'retval' (synthetic) shows its type
                out.append(Statement(node.ea,
                                     f"{vinfo.type} {vinfo.name} = {const_s};"))
                i = end_i
            const_matched = True
            break
        if const_matched:
            continue

        # ── 16-bit negation sequence ──────────────────────────────────────
        neg = _match_neg16(nodes, reg_map, i)
        if neg:
            info, count = neg
            dbg("typesimp", f"  neg16: {info.name} = -{info.name}")
            out.append(Statement(node.ea, f"{info.name} = -{info.name};"))
            i += count
            continue

        # ── Recurse into structured nodes ─────────────────────────────────
        if isinstance(node, IfNode):
            out.append(IfNode(
                ea         = node.ea,
                condition  = _replace_pairs(node.condition, reg_map),
                then_nodes = _simplify(node.then_nodes, reg_map),
                else_nodes = _simplify(node.else_nodes, reg_map),
            ))
            i += 1
            continue

        if isinstance(node, WhileNode):
            out.append(WhileNode(
                ea         = node.ea,
                condition  = _replace_pairs(node.condition, reg_map),
                body_nodes = _simplify(node.body_nodes, reg_map),
            ))
            i += 1
            continue

        if isinstance(node, ForNode):
            out.append(ForNode(
                ea         = node.ea,
                init       = _replace_pairs(node.init,      reg_map),
                condition  = _replace_pairs(node.condition,  reg_map),
                update     = _replace_pairs(node.update,     reg_map),
                body_nodes = _simplify(node.body_nodes, reg_map),
            ))
            i += 1
            continue

        # ── Statement: replace register pairs in text ─────────────────────
        if isinstance(node, Statement):
            new_text = _replace_pairs(node.text, reg_map)
            out.append(Statement(node.ea, new_text) if new_text != node.text else node)
            i += 1
            continue

        # Default pass-through
        out.append(node)
        i += 1

    return out


# ── Pass ──────────────────────────────────────────────────────────────────────

class TypeAwareSimplifier(OptimizationPass):
    """
    Replace register-level expressions with typed variable names and collapse
    common multi-byte patterns (sign tests, negation, …).

    Runs on func.hir after structural passes and HIR assembly.
    No-ops if the function has no known prototype.
    """

    def run(self, func: "Function") -> None:
        from pseudo8051.prototypes import get_proto
        proto = get_proto(func.name)
        if proto is None:
            dbg("typesimp", f"{func.name}: no prototype, skipping")
            return

        reg_map = _build_reg_map(proto)
        if not reg_map:
            dbg("typesimp", f"{func.name}: empty reg map, skipping")
            return

        dbg("typesimp", f"{func.name}: reg_map = {list(reg_map.keys())}")
        func.hir = _simplify(func.hir, reg_map)
