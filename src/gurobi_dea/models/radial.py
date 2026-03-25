"""Classical radial DEA models (CCR and BCC).

References
----------
Charnes, A., Cooper, W.W. & Rhodes, E. (1978). Measuring the efficiency
    of decision making units. European Journal of Operational Research, 2(6), 429-444.
Banker, R.D., Charnes, A. & Cooper, W.W. (1984). Some models for estimating
    technical and scale inefficiencies in DEA. Management Science, 30(9), 1078-1092.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase


class CCR(DEABase):
    """CCR model (Charnes-Cooper-Rhodes, 1978). CRS radial DEA.

    Supports input-oriented and output-oriented formulations.
    Undesirable outputs are treated as inputs (weak disposability).

    Example
    -------
    >>> model = CCR(inputs=["x1"], desirable_outputs=["y1"],
    ...             undesirable_outputs=["b1"], data=df)
    >>> results = model.solve(orientation="input")
    """

    def solve(
        self,
        orientation: Literal["input", "output"] = "input",
        scale: Literal["c", "v"] = "c",
    ) -> pd.DataFrame:
        res = self._empty_results()

        for k in range(self.n_dmu):
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)

            lam = mdl.addVars(self.n_dmu, name="lambda")
            theta = mdl.addVar(name="theta")
            mdl.update()

            if orientation == "input":
                mdl.setObjective(theta, GRB.MINIMIZE)
                # Input constraints: X*lambda <= theta * x_k
                for j in range(self.m):
                    mdl.addConstr(
                        quicksum(self.X[i, j] * lam[i] for i in range(self.n_dmu))
                        <= theta * self.X[k, j]
                    )
                # Desirable output constraints: Y*lambda >= y_k
                for j in range(self.s1):
                    mdl.addConstr(
                        quicksum(self.Y[i, j] * lam[i] for i in range(self.n_dmu))
                        >= self.Y[k, j]
                    )
                # Undesirable outputs treated as inputs
                for j in range(self.s2):
                    mdl.addConstr(
                        quicksum(self.Z[i, j] * lam[i] for i in range(self.n_dmu))
                        <= theta * self.Z[k, j]
                    )
            else:  # output orientation
                mdl.setObjective(theta, GRB.MAXIMIZE)
                for j in range(self.m):
                    mdl.addConstr(
                        quicksum(self.X[i, j] * lam[i] for i in range(self.n_dmu))
                        <= self.X[k, j]
                    )
                for j in range(self.s1):
                    mdl.addConstr(
                        quicksum(self.Y[i, j] * lam[i] for i in range(self.n_dmu))
                        >= theta * self.Y[k, j]
                    )
                for j in range(self.s2):
                    mdl.addConstr(
                        quicksum(self.Z[i, j] * lam[i] for i in range(self.n_dmu))
                        <= self.Z[k, j]
                    )

            # VRS constraint (makes it BCC when scale="v")
            if scale == "v":
                mdl.addConstr(quicksum(lam[i] for i in range(self.n_dmu)) == 1)

            mdl.optimize()
            res.at[k, "TE"] = mdl.objVal if mdl.status == GRB.OPTIMAL else np.nan

        return res


class BCC(CCR):
    """BCC model (Banker-Charnes-Cooper, 1984). VRS radial DEA.

    Convenience wrapper: always uses VRS (scale='v').
    """

    def solve(
        self,
        orientation: Literal["input", "output"] = "input",
        scale: Literal["c", "v"] = "v",
    ) -> pd.DataFrame:
        return super().solve(orientation=orientation, scale="v")
