"""Directional Distance Function (DDF) model.

References
----------
Chambers, R.G., Chung, Y. & Fare, R. (1996). Benefit and distance functions.
    Journal of Economic Theory, 70(2), 407-419.
Chung, Y., Fare, R. & Grosskopf, S. (1997). Productivity and undesirable
    outputs: A directional distance function approach.
    Journal of Environmental Management, 51(3), 229-240.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase


class DDF(DEABase):
    """Directional Distance Function with undesirable outputs.

    Simultaneously expands desirable outputs and contracts inputs
    and undesirable outputs along a specified direction vector.

    Parameters
    ----------
    direction : str
        Direction vector specification:
        - ``'observed'``: use DMU's own (x, y, -b) as direction (default)
        - ``'mean'``: use sample mean as direction
        - ``'unit'``: unit vector (1, 1, ..., 1)

    Example
    -------
    >>> model = DDF(inputs=["x1"], desirable_outputs=["y1"],
    ...             undesirable_outputs=["b1"], data=df)
    >>> results = model.solve(direction="observed")
    """

    def solve(
        self,
        scale: Literal["c", "v"] = "c",
        direction: str = "observed",
    ) -> pd.DataFrame:
        res = self._empty_results()

        for k in range(self.n_dmu):
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)

            beta = mdl.addVar(name="beta", lb=0.0)
            lam = mdl.addVars(self.n_dmu, name="lambda")
            mdl.update()

            # Direction vectors
            if direction == "observed":
                g_x = self.X[k]
                g_y = self.Y[k]
                g_z = self.Z[k]
            elif direction == "mean":
                g_x = self.X.mean(axis=0)
                g_y = self.Y.mean(axis=0)
                g_z = self.Z.mean(axis=0)
            elif direction == "unit":
                g_x = np.ones(self.m)
                g_y = np.ones(self.s1)
                g_z = np.ones(self.s2)
            else:
                raise ValueError(f"Unknown direction: {direction}")

            # Objective: maximize beta
            mdl.setObjective(beta, GRB.MAXIMIZE)

            # Input constraints: X*lambda <= x_k - beta * g_x
            for j in range(self.m):
                mdl.addConstr(
                    quicksum(self.X[i, j] * lam[i] for i in range(self.n_dmu))
                    <= self.X[k, j] - beta * g_x[j]
                )
            # Desirable output: Y*lambda >= y_k + beta * g_y
            for j in range(self.s1):
                mdl.addConstr(
                    quicksum(self.Y[i, j] * lam[i] for i in range(self.n_dmu))
                    >= self.Y[k, j] + beta * g_y[j]
                )
            # Undesirable output: Z*lambda == z_k - beta * g_z
            for j in range(self.s2):
                mdl.addConstr(
                    quicksum(self.Z[i, j] * lam[i] for i in range(self.n_dmu))
                    == self.Z[k, j] - beta * g_z[j]
                )
            # VRS
            if scale == "v":
                mdl.addConstr(quicksum(lam[i] for i in range(self.n_dmu)) == 1)

            mdl.optimize()
            res.at[k, "TE"] = mdl.objVal if mdl.status == GRB.OPTIMAL else np.nan

        return res
