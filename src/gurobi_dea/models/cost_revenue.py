"""Cost, Revenue, and Profit Efficiency models.

Incorporates price information to decompose technical efficiency
into allocative and overall economic efficiency.

References
----------
Fare, R., Grosskopf, S. & Lovell, C.A.K. (1985). The Measurement of
    Efficiency of Production. Kluwer-Nijhoff, Boston.
Tone, K. (2002). A strange case of the cost and allocative efficiencies
    in DEA. Journal of the Operational Research Society, 53(11), 1225-1231.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase


class CostEfficiency(DEABase):
    """Cost efficiency model.

    Minimizes total input cost subject to producing at least the
    observed output levels. Decomposes into:
    - CE (Cost Efficiency) = min_cost / actual_cost
    - TE (Technical Efficiency) from standard DEA
    - AE (Allocative Efficiency) = CE / TE

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs, data, dmu_col :
        Inherited from DEABase.
    input_prices : list[str]
        Column names for input prices (same order as inputs).

    Example
    -------
    >>> model = CostEfficiency(
    ...     inputs=["x1", "x2"], desirable_outputs=["y1"],
    ...     undesirable_outputs=["b1"], data=df,
    ...     input_prices=["w1", "w2"],
    ... )
    >>> results = model.solve(scale="v")
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        input_prices: list[str],
        dmu_col: str = "dmu",
    ) -> None:
        super().__init__(inputs, desirable_outputs, undesirable_outputs, data, dmu_col)
        self.price_names = list(input_prices)
        self.W = data[self.price_names].to_numpy(dtype=float)

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = self._empty_results()
        res["CE"] = np.nan
        res["AE"] = np.nan

        for k in range(self.n_dmu):
            # --- Cost minimization ---
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)
            x_star = mdl.addVars(self.m, name="x_star")
            lam = mdl.addVars(self.n_dmu, name="lambda")
            mdl.update()

            # Minimize cost
            mdl.setObjective(
                quicksum(self.W[k, j] * x_star[j] for j in range(self.m)),
                GRB.MINIMIZE,
            )
            # x_star = X * lambda
            for j in range(self.m):
                mdl.addConstr(
                    quicksum(self.X[i, j] * lam[i] for i in range(self.n_dmu))
                    <= x_star[j]
                )
            # Output constraints
            for j in range(self.s1):
                mdl.addConstr(
                    quicksum(self.Y[i, j] * lam[i] for i in range(self.n_dmu))
                    >= self.Y[k, j]
                )
            for j in range(self.s2):
                mdl.addConstr(
                    quicksum(self.Z[i, j] * lam[i] for i in range(self.n_dmu))
                    <= self.Z[k, j]
                )
            if scale == "v":
                mdl.addConstr(quicksum(lam[i] for i in range(self.n_dmu)) == 1)

            mdl.optimize()

            actual_cost = float(np.dot(self.W[k], self.X[k]))
            if mdl.status == GRB.OPTIMAL and actual_cost > 0:
                min_cost = mdl.objVal
                CE = min_cost / actual_cost
            else:
                CE = np.nan

            # --- Technical efficiency (radial) ---
            mdl2 = gp.Model()
            mdl2.setParam("OutputFlag", 0)
            theta = mdl2.addVar(name="theta")
            lam2 = mdl2.addVars(self.n_dmu, name="lambda")
            mdl2.update()

            mdl2.setObjective(theta, GRB.MINIMIZE)
            for j in range(self.m):
                mdl2.addConstr(
                    quicksum(self.X[i, j] * lam2[i] for i in range(self.n_dmu))
                    <= theta * self.X[k, j]
                )
            for j in range(self.s1):
                mdl2.addConstr(
                    quicksum(self.Y[i, j] * lam2[i] for i in range(self.n_dmu))
                    >= self.Y[k, j]
                )
            for j in range(self.s2):
                mdl2.addConstr(
                    quicksum(self.Z[i, j] * lam2[i] for i in range(self.n_dmu))
                    <= theta * self.Z[k, j]
                )
            if scale == "v":
                mdl2.addConstr(quicksum(lam2[i] for i in range(self.n_dmu)) == 1)

            mdl2.optimize()
            TE = mdl2.objVal if mdl2.status == GRB.OPTIMAL else np.nan

            # Allocative efficiency
            AE = CE / TE if (TE and TE > 0 and not np.isnan(CE)) else np.nan

            res.at[k, "TE"] = TE
            res.at[k, "CE"] = CE
            res.at[k, "AE"] = AE

        return res


class RevenueEfficiency(DEABase):
    """Revenue efficiency model.

    Maximizes total output revenue subject to using no more than
    the observed input levels.

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs, data, dmu_col :
        Inherited from DEABase.
    output_prices : list[str]
        Column names for output prices (same order as desirable_outputs).
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        output_prices: list[str],
        dmu_col: str = "dmu",
    ) -> None:
        super().__init__(inputs, desirable_outputs, undesirable_outputs, data, dmu_col)
        self.price_names = list(output_prices)
        self.P = data[self.price_names].to_numpy(dtype=float)

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = self._empty_results()
        res["RE"] = np.nan
        res["AE"] = np.nan

        for k in range(self.n_dmu):
            # --- Revenue maximization ---
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)
            y_star = mdl.addVars(self.s1, name="y_star")
            lam = mdl.addVars(self.n_dmu, name="lambda")
            mdl.update()

            mdl.setObjective(
                quicksum(self.P[k, j] * y_star[j] for j in range(self.s1)),
                GRB.MAXIMIZE,
            )
            for j in range(self.m):
                mdl.addConstr(
                    quicksum(self.X[i, j] * lam[i] for i in range(self.n_dmu))
                    <= self.X[k, j]
                )
            for j in range(self.s1):
                mdl.addConstr(
                    quicksum(self.Y[i, j] * lam[i] for i in range(self.n_dmu))
                    >= y_star[j]
                )
            for j in range(self.s2):
                mdl.addConstr(
                    quicksum(self.Z[i, j] * lam[i] for i in range(self.n_dmu))
                    <= self.Z[k, j]
                )
            if scale == "v":
                mdl.addConstr(quicksum(lam[i] for i in range(self.n_dmu)) == 1)

            mdl.optimize()

            actual_rev = float(np.dot(self.P[k], self.Y[k]))
            if mdl.status == GRB.OPTIMAL and actual_rev > 0:
                max_rev = mdl.objVal
                RE = actual_rev / max_rev
            else:
                RE = np.nan

            # --- Technical efficiency (output-oriented) ---
            mdl2 = gp.Model()
            mdl2.setParam("OutputFlag", 0)
            phi = mdl2.addVar(name="phi")
            lam2 = mdl2.addVars(self.n_dmu, name="lambda")
            mdl2.update()

            mdl2.setObjective(phi, GRB.MAXIMIZE)
            for j in range(self.m):
                mdl2.addConstr(
                    quicksum(self.X[i, j] * lam2[i] for i in range(self.n_dmu))
                    <= self.X[k, j]
                )
            for j in range(self.s1):
                mdl2.addConstr(
                    quicksum(self.Y[i, j] * lam2[i] for i in range(self.n_dmu))
                    >= phi * self.Y[k, j]
                )
            for j in range(self.s2):
                mdl2.addConstr(
                    quicksum(self.Z[i, j] * lam2[i] for i in range(self.n_dmu))
                    <= self.Z[k, j]
                )
            if scale == "v":
                mdl2.addConstr(quicksum(lam2[i] for i in range(self.n_dmu)) == 1)

            mdl2.optimize()
            TE = 1 / mdl2.objVal if (mdl2.status == GRB.OPTIMAL and mdl2.objVal > 0) else np.nan

            AE = RE / TE if (TE and TE > 0 and not np.isnan(RE)) else np.nan

            res.at[k, "TE"] = TE
            res.at[k, "RE"] = RE
            res.at[k, "AE"] = AE

        return res
