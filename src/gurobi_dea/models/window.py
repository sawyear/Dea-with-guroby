"""Window DEA model.

Applies a sliding window over panel data, treating each DMU-year
observation within the window as a separate unit.

References
----------
Charnes, A., Clark, C.T., Cooper, W.W. & Golany, B. (1985).
    A developmental study of data envelopment analysis in measuring
    the efficiency of maintenance units in the U.S. Air Forces.
    Annals of Operations Research, 2(1), 95-112.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


class WindowDEA:
    """Window DEA for panel data.

    Slides a window of width ``w`` across time periods. Within each
    window, all DMU-year observations form the reference set.

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs : list[str]
    data : pd.DataFrame
        Panel data with 'dmu' and 'year' columns.
    window_width : int
        Number of periods in each window (default 3).
    method : str
        'radial' (default) or 'sbm'.

    Example
    -------
    >>> model = WindowDEA(
    ...     inputs=["x1"], desirable_outputs=["y1"],
    ...     undesirable_outputs=["b1"], data=panel_df,
    ...     window_width=3,
    ... )
    >>> results = model.solve(scale="c")
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        window_width: int = 3,
        method: str = "radial",
    ) -> None:
        self.input_names = list(inputs)
        self.desirable_names = list(desirable_outputs)
        self.undesirable_names = list(undesirable_outputs)
        self.data = data
        self.window_width = window_width
        self.method = method

        self.dmu_ids = sorted(data["dmu"].unique())
        self.years = sorted(data["year"].unique())
        self.m = len(inputs)
        self.s1 = len(desirable_outputs)
        self.s2 = len(undesirable_outputs)

    def _radial_score(self, x_k, y_k, z_k, X_ref, Y_ref, Z_ref, scale):
        n_ref = X_ref.shape[0]
        mdl = gp.Model()
        mdl.setParam("OutputFlag", 0)
        theta = mdl.addVar(name="theta")
        lam = mdl.addVars(n_ref, name="lambda")
        mdl.update()

        mdl.setObjective(theta, GRB.MINIMIZE)
        for j in range(self.m):
            mdl.addConstr(
                quicksum(X_ref[i, j] * lam[i] for i in range(n_ref))
                <= theta * x_k[j]
            )
        for j in range(self.s1):
            mdl.addConstr(
                quicksum(Y_ref[i, j] * lam[i] for i in range(n_ref))
                >= y_k[j]
            )
        for j in range(self.s2):
            mdl.addConstr(
                quicksum(Z_ref[i, j] * lam[i] for i in range(n_ref))
                <= theta * z_k[j]
            )
        if scale == "v":
            mdl.addConstr(quicksum(lam[i] for i in range(n_ref)) == 1)

        mdl.optimize()
        return mdl.objVal if mdl.status == GRB.OPTIMAL else np.nan

    def _sbm_score(self, x_k, y_k, z_k, X_ref, Y_ref, Z_ref, scale):
        n_ref = X_ref.shape[0]
        mdl = gp.Model()
        mdl.setParam("OutputFlag", 0)
        mdl.setParam("NonConvex", 2)

        s_neg = mdl.addVars(self.m, name="s_neg")
        s_pos = mdl.addVars(self.s1, name="s_pos")
        s_und = mdl.addVars(self.s2, name="s_und")
        lam = mdl.addVars(n_ref, name="lambda")
        t = mdl.addVar(name="t")
        mdl.update()

        mdl.setObjective(
            t - (1 / self.m) * quicksum(s_neg[i] / x_k[i] for i in range(self.m)),
            GRB.MINIMIZE,
        )
        for j in range(self.m):
            mdl.addConstr(
                quicksum(X_ref[i, j] * lam[i] for i in range(n_ref))
                == t * x_k[j] - s_neg[j]
            )
        for j in range(self.s1):
            mdl.addConstr(
                quicksum(Y_ref[i, j] * lam[i] for i in range(n_ref))
                == t * y_k[j] + s_pos[j]
            )
        for j in range(self.s2):
            mdl.addConstr(
                quicksum(Z_ref[i, j] * lam[i] for i in range(n_ref))
                == t * z_k[j] - s_und[j]
            )
        denom = self.s1 + self.s2
        mdl.addConstr(
            t + (1 / denom) * (
                quicksum(s_pos[i] / y_k[i] for i in range(self.s1))
                + quicksum(s_und[i] / z_k[i] for i in range(self.s2))
            ) == 1
        )
        if scale == "v":
            mdl.addConstr(quicksum(lam[i] for i in range(n_ref)) == t)

        mdl.optimize()
        return mdl.objVal if mdl.status == GRB.OPTIMAL else np.nan

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        results = []
        w = self.window_width
        indexed = self.data.set_index(["dmu", "year"])

        for start_idx in range(len(self.years) - w + 1):
            window_years = self.years[start_idx: start_idx + w]
            window_label = f"{window_years[0]}-{window_years[-1]}"

            # Build reference set: all DMU-year pairs in window
            mask = self.data["year"].isin(window_years)
            window_data = self.data[mask]
            X_ref = window_data[self.input_names].to_numpy(dtype=float)
            Y_ref = window_data[self.desirable_names].to_numpy(dtype=float)
            Z_ref = window_data[self.undesirable_names].to_numpy(dtype=float)

            # Evaluate each DMU-year in the window
            for yr in window_years:
                for dmu in self.dmu_ids:
                    row = indexed.loc[(dmu, yr)]
                    x_k = row[self.input_names].to_numpy(dtype=float)
                    y_k = row[self.desirable_names].to_numpy(dtype=float)
                    z_k = row[self.undesirable_names].to_numpy(dtype=float)

                    if self.method == "sbm":
                        te = self._sbm_score(x_k, y_k, z_k, X_ref, Y_ref, Z_ref, scale)
                    else:
                        te = self._radial_score(x_k, y_k, z_k, X_ref, Y_ref, Z_ref, scale)

                    results.append({
                        "dmu": dmu,
                        "year": yr,
                        "window": window_label,
                        "TE": te,
                    })

        df_res = pd.DataFrame(results)

        # Average across windows for each DMU-year
        avg = df_res.groupby(["dmu", "year"])["TE"].mean().reset_index()
        avg.rename(columns={"TE": "TE_avg"}, inplace=True)
        df_res = df_res.merge(avg, on=["dmu", "year"])

        return df_res
