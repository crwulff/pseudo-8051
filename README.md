# pseudo8051

An 8051 pseudocode generator for IDA Pro. Decompiles 8051 assembly into readable
C-like pseudocode in a dockable IDA window.

Developed for the RTD2660 display controller (an 8051-variant), but the core
analysis is applicable to any 8051 target.

---

## Usage

1. Open an 8051 binary in IDA Pro.
2. Place the cursor anywhere inside a function.
3. **File → Script File → `run_pseudo8051.py`**

A dockable window titled "8051 Pseudocode-A" (or B, C, … for additional tabs)
opens with the decompiled output. Running the script again on a different function
opens a new tab.

### Interactive features

| Action | Result |
|---|---|
| Double-click a line | Jumps IDA's disassembly view to the corresponding address |
| Right-click → Add local | Declare a named, typed XRAM local variable for this function |
| Right-click → Remove local | Delete an XRAM local declaration |
| Right-click → Toggle hex/dec | Switch numeric constants between hex and decimal |
| Right-click → Show HIR nodes | Toggle IR-level annotation for debugging |

XRAM local variable declarations are stored in IDA netnodes and persist across
sessions in the `.i64` database.

---

## Features

### Structural decompilation

The decompiler recovers high-level control flow from the CFG:

- **Loops** — back-edges are lifted to `while` or `for` nodes; DJNZ loops become
  `for (Rn = N; Rn; Rn--)` when the initial value is known, otherwise
  `while (--Rn != 0)`
- **If/else** — conditional branches are matched against their post-dominator to
  produce `if` / `if-else` nodes; nested structures are handled by iterating
  bottom-up
- **Switch** — the 8051 pattern of cumulative `ADD A, #-N; JZ label` chains is
  recognized and rendered as a `switch` statement

### Type-aware simplification

When a function prototype is available (from IDA's type system, a manual
`PROTOTYPES` entry, or liveness inference), the simplifier:

- Maps registers to named parameters and return variables
- Applies pattern recognition to collapse multi-byte (16- and 32-bit) operations
  into natural C expressions
- Back-propagates type information from callees to clarify register usage in callers

### Pattern library

Multi-byte and idiom patterns recognized and collapsed:

| Pattern | Example output |
|---|---|
| 16-bit addition | `R6R7 += R4R5` |
| 16-bit increment/decrement | `(*ptr)++` |
| 16-bit negation | `R6R7 = -R6R7` |
| Sign extension | `if (A & 0x80) high = 0xFF;` → collapsed |
| XRAM word/dword reads | Consecutive byte reads → `uint16_t` / `uint32_t` load |
| XRAM local writes | Offset-tracked writes to declared locals |
| Multi-byte constant assign | `R4R5R6R7 = 0x12345678` |
| Accumulator relay/fold | Intermediate accumulator copies eliminated |
| Register copy groups | Parallel register moves collapsed |
| Return value | Return register recognized and rendered as `return expr` |

### Analysis passes

- **Constant propagation** (forward dataflow) — propagates known constant values
  through blocks to simplify address calculations and comparisons
- **Liveness analysis** (backward dataflow) — determines register liveness to
  infer function parameters and return registers when no prototype exists

### Expression rendering

Expressions are rendered with correct C operator precedence; parentheses are added
only where needed. All standard C operators are supported including compound
assignment (`+=`, `|=`, etc.), unary operators, casts, and pointer dereferences.

---

## XRAM local variables

Functions frequently use XRAM scratch space as local storage. You can declare
named typed locals on a per-function basis; these are shown in the pseudocode
output and persist in the IDA database.

```python
# In IDA's Python console:
import pseudo8051
pseudo8051.set_local(here(), 0xdc8a, 'count', 'int16_t')
pseudo8051.list_locals(here())
pseudo8051.del_local(here(), 0xdc8a)
```

The right-click menu in the pseudocode window provides the same operations
interactively.

---

## Function prototypes

Prototypes are resolved in priority order:

1. Manual entry in `pseudo8051/prototypes.py`
2. IDA's type system (set with the **y** key in the disassembly)
3. Liveness inference (registers live at entry / return are treated as
   parameters / return value)

Standard 8051 calling convention: return value in `A` (bool/uint8\_t),
`R6:R7` (uint16\_t), or `R4:R5:R6:R7` (uint32\_t); parameters allocated
from R7 downward.

---

## Project structure

```
pseudo8051_ida/
├── run_pseudo8051.py        IDA entry point (reload-safe)
├── pseudo8051/
│   ├── __init__.py          PseudocodeViewer, run_pseudocode_view()
│   ├── constants.py         Register IDs, SFR names, debug flag
│   ├── prototypes.py        Function prototype management
│   ├── locals.py            XRAM local variable storage (netnodes)
│   ├── locals_ui.py         Right-click menu UI
│   ├── ir/
│   │   ├── expr.py          Expression tree (Reg, Const, BinOp, XRAMRef, …)
│   │   ├── hir.py           HIR nodes (Assign, IfGoto, IfNode, WhileNode, …)
│   │   ├── basicblock.py    Basic block + instruction lifting
│   │   ├── function.py      Function graph, pass runner, renderer
│   │   ├── instruction.py   IDA instruction wrapper + handler dispatch
│   │   ├── operand.py       IDA operand → Expr conversion
│   │   └── cpstate.py       Constant propagation state
│   ├── handlers/            Per-mnemonic lift functions (mov, arith, logic, …)
│   ├── analysis/
│   │   ├── constprop.py     Forward constant propagation
│   │   └── liveness.py      Backward liveness analysis
│   └── passes/
│       ├── rmw.py           XRAM read-modify-write collapser
│       ├── loops.py         Loop structurer
│       ├── switch.py        Switch structurer
│       ├── ifelse.py        If/else structurer
│       ├── typesimplify.py  Type-aware simplifier
│       └── patterns/        Individual pattern modules
└── tests/
    ├── conftest.py          IDA module mocks (no IDA install needed)
    ├── helpers.py           Test utilities
    ├── test_pipeline.py     End-to-end tests
    ├── test_patterns.py     Pattern unit tests
    └── test_typesimplify.py Simplifier tests
```

---

## Running tests

No IDA installation required — `conftest.py` provides mock IDA modules.

```bash
pytest tests/
```

---

## Live reloading

`run_pseudo8051.py` re-imports all modules in dependency order on every
invocation. You can edit any source file and re-run the script to pick up
changes immediately, without restarting IDA.

---

## License

MIT — see [LICENSE](LICENSE).
