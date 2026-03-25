"""
passes/patterns — Pattern ABC and built-in pattern registry.

To add a new pattern:
  1. Create a new .py file in this directory.
  2. Subclass Pattern, implement match(), and add a docstring.
  3. Import the class below and append an instance to _PATTERNS.
"""

from typing import List

from pseudo8051.passes.patterns.base             import Pattern, Match, Simplify       # noqa: F401
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
from pseudo8051.passes.patterns.accum_relay      import AccumRelayPattern
from pseudo8051.passes.patterns.accum_fold       import AccumFoldPattern

_PATTERNS: List[Pattern] = [
    SignBitTestPattern(),
    RetvalPattern(),           # rename call return → retvalN; updates reg_map
    RegCopyGroupPattern(),     # propagate retval across reg copies; drops copy stmts
    AccumRelayPattern(),       # collapse A=expr; target=A; → target=expr;
    AccumFoldPattern(),        # collapse A-chain + IfGoto/IfNode/Assign terminal
    MultiByteAddPattern(),     # before XRAMLocalWrite: consumes the whole ADD+ADDC sequence
    MultiByteIncDecPattern(),  # collapse multi-byte inc/dec chains into var++/var--
    IfNodeIncDecPattern(),     # IfNode carry form (post-loop-structuring)
    XRAMLocalWritePattern(),   # before ConstGroup so locals are handled first
    ConstGroupPattern(),
    XRAMGroupReadPattern(),
    Neg16Pattern(),
]
