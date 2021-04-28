"""Fixes for BCC-ESM1 model."""
from ..common import ClFixHybridPressureCoord
from .bcc_csm2_mr import Siconc as BaseSiconc
from .bcc_csm2_mr import Sos as BaseSos
from .bcc_csm2_mr import Tos as BaseTos


Cl = ClFixHybridPressureCoord


Cli = ClFixHybridPressureCoord


Clw = ClFixHybridPressureCoord


Siconc = BaseSiconc


Sos = BaseSos


Tos = BaseTos
