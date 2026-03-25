"""gurobi-dea: Modular DEA toolkit powered by Gurobi.

Supported models:
    Static (cross-sectional):
        - CCR / BCC (radial, Charnes-Cooper-Rhodes / Banker-Charnes-Cooper)
        - SBM (Slacks-Based Measure) with undesirable outputs
        - Super-SBM (Super-efficiency)
        - EBM (Epsilon-Based Measure)
        - Additive model
        - DDF (Directional Distance Function)

    Structural:
        - Network SBM / Network EBM (up to 3 stages)
        - Dynamic SBM (carry-over)

    Productivity indices:
        - Malmquist TFP Index
        - Malmquist-Luenberger Index (environmental TFP)
"""

from gurobi_dea.models.radial import CCR, BCC
from gurobi_dea.models.sbm import SBM, SuperSBM
from gurobi_dea.models.ebm import EBM
from gurobi_dea.models.additive import Additive
from gurobi_dea.models.ddf import DDF
from gurobi_dea.models.network import NetworkSBM, NetworkEBM
from gurobi_dea.models.dynamic import DynamicSBM
from gurobi_dea.models.malmquist import Malmquist
from gurobi_dea.models.malmquist_luenberger import MalmquistLuenberger

__version__ = "0.3.0"
__all__ = [
    "CCR", "BCC",
    "SBM", "SuperSBM",
    "EBM",
    "Additive",
    "DDF",
    "NetworkSBM", "NetworkEBM",
    "DynamicSBM",
    "Malmquist",
    "MalmquistLuenberger",
]
