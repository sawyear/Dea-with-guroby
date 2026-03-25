"""gurobi-dea: Modular DEA toolkit powered by Gurobi.

Supported models:
    - SBM (Slacks-Based Measure) with undesirable outputs
    - Super-SBM (Super-efficiency)
    - EBM (Epsilon-Based Measure)
    - ADD (Additive model)
    - Network SBM / Network EBM (up to 3 stages)
    - Dynamic SBM (carry-over)
"""

from gurobi_dea.models.sbm import SBM, SuperSBM
from gurobi_dea.models.ebm import EBM
from gurobi_dea.models.additive import Additive
from gurobi_dea.models.network import NetworkSBM, NetworkEBM
from gurobi_dea.models.dynamic import DynamicSBM

__version__ = "0.2.0"
__all__ = ["SBM", "SuperSBM", "EBM", "Additive", "NetworkSBM", "NetworkEBM", "DynamicSBM"]
