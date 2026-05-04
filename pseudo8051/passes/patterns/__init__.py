"""
passes/patterns — Transform hierarchy and built-in transform registry.

To add a new transform:
  1. Create a new .py file in this directory.
  2. Subclass the appropriate typed base class (SubstituteTransform,
     InlineTransform, CombineTransform, EliminateTransform, or
     RestructureTransform), implement the required method, and add a docstring.
  3. Import the class below and append an instance to _PATTERNS.
"""

from typing import List

from pseudo8051.passes.patterns.base             import (                               # noqa: F401
    Transform, Pattern, Match, Simplify,
    SubstituteTransform, InlineTransform, CombineTransform,
    EliminateTransform, RestructureTransform,
)
from pseudo8051.passes.patterns._utils           import VarInfo, _replace_pairs        # noqa: F401
from pseudo8051.passes.patterns.sign_bit         import SignBitTestPattern
from pseudo8051.passes.patterns.neg16            import Neg16Pattern
from pseudo8051.passes.patterns.const_group      import ConstGroupPattern
from pseudo8051.passes.patterns.xram_group_read  import XRAMGroupReadPattern
from pseudo8051.passes.patterns.xram_local_write import XRAMLocalWritePattern
from pseudo8051.passes.patterns.mb_add           import MultiByteAddPattern
from pseudo8051.passes.patterns.mb_incdec        import MultiByteIncDecPattern, IfNodeIncDecPattern
from pseudo8051.passes.patterns.retval           import RetvalPattern
from pseudo8051.passes.patterns.reg_copy_group   import RegCopyGroupPattern
from pseudo8051.passes.patterns.reg_inc          import RegPostIncPattern, RegPreIncPattern
from pseudo8051.passes.patterns.accum_relay      import AccumRelayPattern
from pseudo8051.passes.patterns.mul16            import Mul16Pattern
from pseudo8051.passes.patterns.rol_switch       import RolSwitchPattern
from pseudo8051.passes.patterns.accum_fold       import AccumFoldPattern
from pseudo8051.passes.patterns.xch_copy         import XchCopyPattern
from pseudo8051.passes.patterns.zero_ext_group   import ZeroExtGroupPattern

_PATTERNS: List[Pattern] = [
    SignBitTestPattern(),
    RetvalPattern(),           # rename call return → retvalN; updates reg_map
    RegCopyGroupPattern(),     # propagate retval across reg copies; drops copy stmts
    XchCopyPattern(),          # before RegPostInc: handles multi-DPTR++ XCH-swap ROM→XRAM copy
    RegPostIncPattern(),       # any node using Rn once + Rn++/-- → embed post-op
    RegPreIncPattern(),        # Rn++/-- + any node using Rn once → embed pre-op
    AccumRelayPattern(),       # collapse A=expr; target=A; → target=expr;
    RolSwitchPattern(),        # collapse A=rol8(A)×N + compounds before switch(A>>K)
    Mul16Pattern(),            # collapse 16×16→16 multiply idiom into {A,Rlo1} = pair1 * pair2
    AccumFoldPattern(),        # collapse A-chain + IfGoto/IfNode/Assign terminal
    MultiByteAddPattern(),     # before XRAMLocalWrite: consumes the whole ADD+ADDC sequence
    MultiByteIncDecPattern(),  # collapse multi-byte inc/dec chains into var++/var--
    IfNodeIncDecPattern(),     # IfNode carry form (post-loop-structuring)
    XRAMLocalWritePattern(),   # before ConstGroup so locals are handled first
    ConstGroupPattern(),
    ZeroExtGroupPattern(),
    XRAMGroupReadPattern(),
    Neg16Pattern(),
]
