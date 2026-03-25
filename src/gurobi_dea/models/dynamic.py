"""Dynamic SBM-DEA with carry-over variables.

References
----------
Tone, K. & Tsutsui, M. (2010). Dynamic DEA: A slacks-based measure approach.
    Omega, 38(3-4), 145-156.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


class DynamicSBM:
    """Dynamic SBM-DEA with carry-over and undesirable outputs.

    Parameters
    ----------
    inputs : list[str]
        Input variable column names.
    desirable_outputs : list[str]
        Desirable output column names.
    undesirable_outputs : list[str]
        Undesirable output column names.
    carry_overs : list[str]
        Carry-over (link) variable column names between periods.
    data : pd.DataFrame
        Panel data with columns ``'dmu'``, ``'year'``, and all variables.
    weight_dmu : list[float] | None
        DMU weights (default: equal).
    weight_year : list[float] | None
        Year weights (default: equal).

    Example
    -------
    >>> model = DynamicSBM(
    ...     inputs=["x1"], desirable_outputs=["y1"],
    ...     undesirable_outputs=["b1"], carry_overs=["k1"],
    ...     data=panel_df,
    ... )
    >>> results = model.solve(scale="c")
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        carry_overs: list[str],
        data: pd.DataFrame,
        weight_dmu: list[float] | None = None,
        weight_year: list[float] | None = None,
    ) -> None:
        self.input_names = list(inputs)
        self.desirable_names = list(desirable_outputs)
        self.undesirable_names = list(undesirable_outputs)
        self.carry_names = list(carry_overs)
        self.data = data

        self.dmu_ids = data["dmu"].unique()
        self.years = sorted(data["year"].unique())
        self.n_dmu = len(self.dmu_ids)
        self.n_year = len(self.years)

        self.m = len(inputs)
        self.s1 = len(desirable_outputs)
        self.s2 = len(undesirable_outputs)
        self.s3 = len(carry_overs)

        self.w_dmu = np.array(weight_dmu) if weight_dmu else np.ones(self.n_dmu) / self.n_dmu
        self.w_year = np.array(weight_year) if weight_year else np.ones(self.n_year) / self.n_year

        # Index data by (dmu, year) for fast lookup
        self._indexed = data.set_index(["dmu", "year"])

    def _get(self, dmu, year, cols: list[str]) -> np.ndarray:
        return self._indexed.loc[(dmu, year), cols].to_numpy(dtype=float)

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = pd.DataFrame(
            columns=["dmu", "TE"] + [str(y) for y in self.years],
            index=range(self.n_dmu),
        )
        res["dmu"] = self.dmu_ids

        for k_idx, k_dmu in enumerate(self.dmu_ids):
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)
            mdl.setParam("NonConvex", 2)

            # Variables indexed by (var_idx, year_idx)
            s_neg = mdl.addVars(self.m, self.n_year, name="s_neg")
            s_pos = mdl.addVars(self.s1, self.n_year, name="s_pos")
            s_und = mdl.addVars(self.s2, self.n_year, name="s_und")
            s_carry = mdl.addVars(self.s3, self.n_year, name="s_carry")
            lam = mdl.addVars(self.n_dmu, self.n_year, name="lambda")
            t = mdl.addVar(name="t")
            y_k = mdl.addVars(self.n_year, name="y_k")
            x_k = mdl.addVars(self.n_year, name="x_k")
            mdl.update()

            for yr_idx, yr in enumerate(self.years):
                # Data matrices for this year
                X_yr = np.array([self._get(d, yr, self.input_names) for d in self.dmu_ids])
                Y_yr = np.array([self._get(d, yr, self.desirable_names) for d in self.dmu_ids])
                Z_yr = np.array([self._get(d, yr, self.undesirable_names) for d in self.dmu_ids])
                C_yr = np.array([self._get(d, yr, self.carry_names) for d in self.dmu_ids])

                x_k0 = X_yr[k_idx]
                y_k0 = Y_yr[k_idx]
                z_k0 = Z_yr[k_idx]
                c_k0 = C_yr[k_idx]

                # Input constraints
                for j in range(self.m):
                    mdl.addConstr(
                        quicksum(X_yr[i, j] * lam[i, yr_idx] for i in range(self.n_dmu))
                        == t * x_k0[j] - s_neg[j, yr_idx]
                    )
                # Desirable output constraints
                for j in range(self.s1):
                    mdl.addConstr(
                        quicksum(Y_yr[i, j] * lam[i, yr_idx] for i in range(self.n_dmu))
                        == t * y_k0[j] + s_pos[j, yr_idx]
                    )
                # Undesirable output constraints
                for j in range(self.s2):
                    mdl.addConstr(
                        quicksum(Z_yr[i, j] * lam[i, yr_idx] for i in range(self.n_dmu))
                        == t * z_k0[j] - s_und[j, yr_idx]
                    )
                # Carry-over constraints
                for j in range(self.s3):
                    mdl.addConstr(
                        quicksum(C_yr[i, j] * lam[i, yr_idx] for i in range(self.n_dmu))
                        == t * c_k0[j] - s_carry[j, yr_idx]
                    )
                # Carry-over link between periods
                if yr_idx + 1 < self.n_year:
                    C_next = np.array([
                        self._get(d, self.years[yr_idx + 1], self.carry_names)
                        for d in self.dmu_ids
                    ])
                    for j in range(self.s3):
                        mdl.addConstr(
                            quicksum(C_yr[i, j] * lam[i, yr_idx] for i in range(self.n_dmu))
                            == quicksum(C_next[i, j] * lam[i, yr_idx + 1] for i in range(self.n_dmu))
                        )
                # VRS
                if scale == "v":
                    mdl.addConstr(quicksum(lam[i, yr_idx] for i in range(self.n_dmu)) == t)

                # Per-year efficiency components
                denom_out = self.s1 + self.s2
                mdl.addConstr(
                    y_k[yr_idx] == t + (1 / denom_out) * (
                        quicksum(s_pos[j, yr_idx] / y_k0[j] for j in range(self.s1))
                        + quicksum(s_und[j, yr_idx] / z_k0[j] for j in range(self.s2))
                    )
                )
                denom_in = self.m + self.s3
                mdl.addConstr(
                    x_k[yr_idx] == t - (1 / denom_in) * (
                        quicksum(s_neg[j, yr_idx] / x_k0[j] for j in range(self.m))
                        + quicksum(s_carry[j, yr_idx] / c_k0[j] for j in range(self.s3))
                    )
                )

            # Normalization
            mdl.addConstr(
                quicksum(float(self.w_year[yr]) * y_k[yr] for yr in range(self.n_year)) == 1
            )
            # Objective
            mdl.setObjective(
                quicksum(float(self.w_year[yr]) * x_k[yr] for yr in range(self.n_year)),
                GRB.MINIMIZE,
            )

            mdl.optimize()

            res.at[k_idx, "TE"] = mdl.objVal
            for yr_idx in range(self.n_year):
                res.loc[k_idx, str(self.years[yr_idx])] = x_k[yr_idx].X

        return res
