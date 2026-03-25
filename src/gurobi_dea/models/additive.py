"""Additive DEA model.

References
----------
Charnes, A., Cooper, W.W., Golany, B., Seiford, L., & Stutz, J. (1985).
    Foundations of data envelopment analysis for Pareto-Koopmans
    efficient empirical production functions.
    Journal of Econometrics, 30(1-2), 91-107.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase


class Additive(DEABase):
    """Additive DEA with undesirable outputs.

    Maximizes the weighted sum of normalized input and output slacks.
    """

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = self._empty_results()

        for k in range(self.n_dmu):
            m = gp.Model()
            m.setParam("OutputFlag", 0)
            m.setParam("NonConvex", 2)

            s_neg = m.addVars(self.m, name="s_neg")
            s_pos = m.addVars(self.s1, name="s_pos")
            s_und = m.addVars(self.s2, name="s_und")
            lam = m.addVars(self.n_dmu, name="lambda")
            m.update()

            # Objective: maximize normalized slacks
            m.setObjective(
                quicksum(s_neg[i] / self.X[k, i] for i in range(self.m))
                + quicksum(s_pos[i] / self.Y[k, i] for i in range(self.s1)),
                GRB.MAXIMIZE,
            )

            # Input constraints
            for j in range(self.m):
                m.addConstr(
                    quicksum(self.X[i, j] * lam[i] for i in range(self.n_dmu))
                    == self.X[k, j] - s_neg[j]
                )
            # Desirable output constraints
            for j in range(self.s1):
                m.addConstr(
                    quicksum(self.Y[i, j] * lam[i] for i in range(self.n_dmu))
                    == self.Y[k, j] + s_pos[j]
                )
            # Undesirable output constraints
            for j in range(self.s2):
                m.addConstr(
                    quicksum(self.Z[i, j] * lam[i] for i in range(self.n_dmu))
                    == self.Z[k, j] - s_und[j]
                )
            # VRS
            if scale == "v":
                m.addConstr(quicksum(lam[i] for i in range(self.n_dmu)) == 1)

            m.optimize()

            res.at[k, "TE"] = m.objVal
            for i in range(self.m):
                res.loc[k, self.input_names[i]] = s_neg[i].X
            for i in range(self.s1):
                res.loc[k, self.desirable_names[i]] = s_pos[i].X
            for i in range(self.s2):
                res.loc[k, self.undesirable_names[i]] = s_und[i].X

        return res
