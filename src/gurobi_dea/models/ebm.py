"""EBM (Epsilon-Based Measure) model.

References
----------
Tone, K. & Tsutsui, M. (2010). An epsilon-based measure of efficiency
    in DEA - A third pole of technical efficiency.
    European Journal of Operational Research, 207(3), 1554-1563.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase
from gurobi_dea.models.sbm import SBM
from gurobi_dea.utils import affinity_matrix


class EBM(DEABase):
    """EBM-DEA with undesirable outputs.

    Automatically computes epsilon and weights from the affinity matrix
    of SBM slacks (VRS reference).

    Example
    -------
    >>> model = EBM(
    ...     inputs=["x1", "x2"],
    ...     desirable_outputs=["y1"],
    ...     undesirable_outputs=["b1"],
    ...     data=df,
    ... )
    >>> results = model.solve(scale="c")
    """

    def _compute_affinity_params(self) -> None:
        """Run SBM(VRS) to get slacks, then compute affinity matrices."""
        sbm = SBM(
            inputs=self.input_names,
            desirable_outputs=self.desirable_names,
            undesirable_outputs=self.undesirable_names,
            data=self.data,
            dmu_col=self.dmu_col,
        )
        sbm_res = sbm.solve(scale="v")

        # Input slacks
        slack_x = self.X - sbm_res[self.input_names].to_numpy(dtype=float)
        self.eps_x, self.w_x = affinity_matrix(slack_x)

        # Desirable output slacks
        slack_y = sbm_res[self.desirable_names].to_numpy(dtype=float) + self.Y
        self.eps_y, self.w_y = affinity_matrix(slack_y)

        # Undesirable output slacks
        slack_z = self.Z - sbm_res[self.undesirable_names].to_numpy(dtype=float)
        self.eps_z, self.w_z = affinity_matrix(slack_z)

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        self._compute_affinity_params()
        res = self._empty_results()

        for k in range(self.n_dmu):
            m = gp.Model()
            m.setParam("OutputFlag", 0)
            m.setParam("NonConvex", 2)

            theta = m.addVar(name="theta")
            eta = m.addVar(name="eta")
            s_neg = m.addVars(self.m, name="s_neg")
            s_pos = m.addVars(self.s1, name="s_pos")
            s_und = m.addVars(self.s2, name="s_und")
            lam = m.addVars(self.n_dmu, name="lambda")
            m.update()

            # Objective
            m.setObjective(
                theta
                - self.eps_x * quicksum(
                    float(self.w_x[i]) * s_neg[i] / self.X[k, i]
                    for i in range(self.m)
                ),
                GRB.MINIMIZE,
            )

            # Input constraints
            for j in range(self.m):
                m.addConstr(
                    quicksum(self.X[i, j] * lam[i] for i in range(self.n_dmu))
                    == theta * self.X[k, j] - s_neg[j]
                )
            # Desirable output constraints
            for j in range(self.s1):
                m.addConstr(
                    quicksum(self.Y[i, j] * lam[i] for i in range(self.n_dmu))
                    == eta * self.Y[k, j] + s_pos[j]
                )
            # Undesirable output constraints
            for j in range(self.s2):
                m.addConstr(
                    quicksum(self.Z[i, j] * lam[i] for i in range(self.n_dmu))
                    == eta * self.Z[k, j] - s_und[j]
                )
            # Normalization
            m.addConstr(
                eta
                + self.eps_y * quicksum(
                    float(self.w_y[i]) * s_pos[i] / self.Y[k, i]
                    for i in range(self.s1)
                )
                + self.eps_z * quicksum(
                    float(self.w_z[i]) * s_und[i] / self.Z[k, i]
                    for i in range(self.s2)
                )
                == 1
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
