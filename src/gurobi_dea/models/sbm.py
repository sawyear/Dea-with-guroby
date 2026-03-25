"""SBM (Slacks-Based Measure) and Super-SBM models.

References
----------
Tone, K. (2001). A slacks-based measure of efficiency in DEA.
    European Journal of Operational Research, 130(3), 498-509.
Tone, K. (2002). A slacks-based measure of super-efficiency in DEA.
    European Journal of Operational Research, 143(1), 32-41.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase


class SBM(DEABase):
    """SBM-DEA with undesirable outputs (Tone, 2001).

    Uses the Charnes-Cooper transformation to linearize the fractional
    program.

    Example
    -------
    >>> model = SBM(
    ...     inputs=["x1", "x2"],
    ...     desirable_outputs=["y1"],
    ...     undesirable_outputs=["b1"],
    ...     data=df,
    ... )
    >>> results = model.solve(scale="c")
    """

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = self._empty_results()

        for k in range(self.n_dmu):
            m = gp.Model()
            m.setParam("OutputFlag", 0)
            m.setParam("NonConvex", 2)

            # Decision variables
            s_neg = m.addVars(self.m, name="s_neg")          # input slacks
            s_pos = m.addVars(self.s1, name="s_pos")         # desirable output slacks
            s_und = m.addVars(self.s2, name="s_und")         # undesirable output slacks
            lam = m.addVars(self.n_dmu, name="lambda")       # intensity
            t = m.addVar(name="t")                           # C-C scalar
            m.update()

            # Objective (after C-C transformation)
            m.setObjective(
                t - (1 / self.m) * quicksum(
                    s_neg[i] / self.X[k, i] for i in range(self.m)
                ),
                GRB.MINIMIZE,
            )

            # Input constraints
            for j in range(self.m):
                m.addConstr(
                    quicksum(self.X[i, j] * lam[i] for i in range(self.n_dmu))
                    == t * self.X[k, j] - s_neg[j]
                )
            # Desirable output constraints
            for j in range(self.s1):
                m.addConstr(
                    quicksum(self.Y[i, j] * lam[i] for i in range(self.n_dmu))
                    == t * self.Y[k, j] + s_pos[j]
                )
            # Undesirable output constraints
            for j in range(self.s2):
                m.addConstr(
                    quicksum(self.Z[i, j] * lam[i] for i in range(self.n_dmu))
                    == t * self.Z[k, j] - s_und[j]
                )
            # Normalization
            denom = self.s1 + self.s2
            m.addConstr(
                t
                + (1 / denom) * (
                    quicksum(s_pos[i] / self.Y[k, i] for i in range(self.s1))
                    + quicksum(s_und[i] / self.Z[k, i] for i in range(self.s2))
                )
                == 1
            )
            # Returns to scale
            if scale == "v":
                m.addConstr(quicksum(lam[i] for i in range(self.n_dmu)) == t)

            m.optimize()

            # Store results
            t_val = t.X if t.X != 0 else 1.0
            res.at[k, "TE"] = m.objVal
            for i in range(self.m):
                res.loc[k, self.input_names[i]] = s_neg[i].X / t_val
            for i in range(self.s1):
                res.loc[k, self.desirable_names[i]] = s_pos[i].X / t_val
            for i in range(self.s2):
                res.loc[k, self.undesirable_names[i]] = s_und[i].X / t_val

        return res


class SuperSBM(DEABase):
    """Super-efficiency SBM (Tone, 2002).

    Excludes the evaluated DMU from the reference set.
    """

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = self._empty_results()

        for k in range(self.n_dmu):
            m = gp.Model()
            m.setParam("OutputFlag", 0)
            m.setParam("NonConvex", 2)

            fi = m.addVars(self.m, name="fi")
            fo = m.addVars(self.s1, name="fo")
            sigma = m.addVars(self.s2, name="sigma")
            lam = m.addVars(self.n_dmu, name="lambda")
            t = m.addVar(name="t")
            m.update()

            # Objective
            m.setObjective(
                t + (1 / self.m) * quicksum(fi[j] for j in range(self.m)),
                GRB.MINIMIZE,
            )

            # Constraints (exclude DMU k from reference set)
            ref = [i for i in range(self.n_dmu) if i != k]
            for j in range(self.m):
                m.addConstr(
                    quicksum(lam[i] * self.X[i, j] for i in ref)
                    <= (1 + fi[j]) * self.X[k, j]
                )
            for j in range(self.s1):
                m.addConstr(
                    quicksum(lam[i] * self.Y[i, j] for i in ref)
                    >= (1 - fo[j]) * self.Y[k, j]
                )
            for j in range(self.s2):
                m.addConstr(
                    quicksum(lam[i] * self.Z[i, j] for i in ref)
                    <= (1 + sigma[j]) * self.Z[k, j]
                )
            # Normalization
            m.addConstr(
                t - (t / self.s1) * quicksum(fo[j] for j in range(self.s1)) == 1
            )
            # VRS
            if scale == "v":
                m.addConstr(quicksum(lam[i] for i in ref) == 1)

            m.optimize()

            t_val = t.X if t.X != 0 else 1.0
            res.at[k, "TE"] = m.objVal
            for i in range(self.m):
                res.loc[k, self.input_names[i]] = fi[i].X / t_val
            for i in range(self.s1):
                res.loc[k, self.desirable_names[i]] = fo[i].X / t_val
            for i in range(self.s2):
                res.loc[k, self.undesirable_names[i]] = sigma[i].X / t_val

        return res
