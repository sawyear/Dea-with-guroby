"""gurobi-dea: Modular DEA toolkit powered by Gurobi.

Supported models:
    Static (cross-sectional):
        - CCR / BCC (radial)
        - SBM / Super-SBM (slacks-based)
        - EBM (epsilon-based)
        - Additive
        - DDF (directional distance function)
        - Cross-efficiency (peer appraisal)
        - Meta-frontier (technology gap ratio)
        - Bootstrap DEA (bias correction + CI)
        - Cost / Revenue efficiency (allocative decomposition)

    Structural:
        - Network SBM / Network EBM
        - Dynamic SBM
        - Dynamic Network SBM

    Panel:
        - Window DEA

    Productivity indices:
        - Malmquist TFP Index
        - Malmquist-Luenberger Index
"""

from gurobi_dea.models.radial import CCR, BCC
from gurobi_dea.models.sbm import SBM, SuperSBM
from gurobi_dea.models.ebm import EBM
from gurobi_dea.models.additive import Additive
from gurobi_dea.models.ddf import DDF
from gurobi_dea.models.cross_efficiency import CrossEfficiency
from gurobi_dea.models.meta_frontier import MetaFrontier
from gurobi_dea.models.bootstrap import BootstrapDEA
from gurobi_dea.models.window import WindowDEA
from gurobi_dea.models.cost_revenue import CostEfficiency, RevenueEfficiency
from gurobi_dea.models.network import NetworkSBM, NetworkEBM
from gurobi_dea.models.dynamic import DynamicSBM
from gurobi_dea.models.dynamic_network import DynamicNetworkSBM
from gurobi_dea.models.malmquist import Malmquist
from gurobi_dea.models.malmquist_luenberger import MalmquistLuenberger

__version__ = "0.5.0"
__all__ = [
    "CCR", "BCC",
    "SBM", "SuperSBM",
    "EBM",
    "Additive",
    "DDF",
    "CrossEfficiency",
    "MetaFrontier",
    "BootstrapDEA",
    "WindowDEA",
    "CostEfficiency", "RevenueEfficiency",
    "NetworkSBM", "NetworkEBM",
    "DynamicSBM",
    "DynamicNetworkSBM",
    "Malmquist",
    "MalmquistLuenberger",
]
