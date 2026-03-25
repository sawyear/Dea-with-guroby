"""Malmquist-Luenberger Productivity Index.

Based on directional distance functions, designed for settings with
undesirable outputs (e.g., pollution, CO2 emissions).

References
----------
Chung, Y., Fare, R. & Grosskopf, S. (1997). Productivity and undesirable
    outputs: A directional distance function approach.
    Journal of Environmental Management, 51(3), 229-240.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


class MalmquistLuenberger:
    """Malmquist-Luenberger (ML) index for environmental TFP.

    Uses DDF to handle undesirable outputs properly.
    ML = EC_ml * TC_ml, where values > 1 indicate improvement.

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs : list[str]
    data : pd.DataFrame
        Panel data with 'dmu' and 'year' columns.
    direction : str
        'observed' (default) or 'mean'.

    Example
    -------
    >>> ml = MalmquistLuenberger(
    ...     inputs=["x1"], desirable_outputs=["y1"],
    ...     undesirable_outputs=["co2"], data=panel_df,
    ... )
    >>> results = ml.compute()
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        direction: str = "observed",
    ) -> None:
        self.input_names = list(inputs)
        self.desirable_names = list(desirable_outputs)
        self.undesirable_names = list(undesirable_outputs)
        self.data = data
        self.direction = direction
        self.dmu_ids = sorted(data["dmu"].unique())
        self.years = sorted(data["year"].unique())
        self._indexed = data.set_index(["dmu", "year"])

    def _get_period_data(self, year):
        mask = self.data["year"] == year
        d = self.data[mask].sort_values("dmu")
        X = d[self.input_names].to_numpy(dtype=float)
        Y = d[self.desirable_names].to_numpy(dtype=float)
        Z = d[self.undesirable_names].to_numpy(dtype=float)
        return X, Y, Z

    def _get_dmu_data(self, dmu, year):
        row = self._indexed.loc[(dmu, year)]
        return (
            row[self.input_names].to_numpy(dtype=float),
            row[self.desirable_names].to_numpy(dtype=float),
            row[self.undesirable_names].to_numpy(dtype=float),
        )

    def _ddf_distance(
        self,
        x_k, y_k, z_k,
        X_ref, Y_ref, Z_ref,
        g_x, g_y, g_z,
    ) -> float:
        """Solve DDF: max beta s.t. technology constraints."""
        n_ref = X_ref.shape[0]
        m, s1, s2 = X_ref.shape[1], Y_ref.shape[1], Z_ref.shape[1]

        mdl = gp.Model()
        mdl.setParam("OutputFlag", 0)
        beta = mdl.addVar(name="beta", lb=-GRB.INFINITY)
        lam = mdl.addVars(n_ref, name="lambda")
        mdl.update()

        mdl.setObjective(beta, GRB.MAXIMIZE)

        for j in range(m):
            mdl.addConstr(
                quicksum(X_ref[i, j] * lam[i] for i in range(n_ref))
                <= x_k[j] - beta * g_x[j]
            )
        for j in range(s1):
            mdl.addConstr(
                quicksum(Y_ref[i, j] * lam[i] for i in range(n_ref))
                >= y_k[j] + beta * g_y[j]
            )
        for j in range(s2):
            mdl.addConstr(
                quicksum(Z_ref[i, j] * lam[i] for i in range(n_ref))
                == z_k[j] - beta * g_z[j]
            )

        mdl.optimize()
        if mdl.status == GRB.OPTIMAL:
            return mdl.objVal
        return np.nan

    def _get_direction(self, x_k, y_k, z_k):
        if self.direction == "observed":
            return x_k.copy(), y_k.copy(), z_k.copy()
        elif self.direction == "mean":
            X_all = self.data[self.input_names].to_numpy(dtype=float)
            Y_all = self.data[self.desirable_names].to_numpy(dtype=float)
            Z_all = self.data[self.undesirable_names].to_numpy(dtype=float)
            return X_all.mean(0), Y_all.mean(0), Z_all.mean(0)
        raise ValueError(f"Unknown direction: {self.direction}")

    def compute(self) -> pd.DataFrame:
        """Compute ML index for consecutive year pairs.

        Returns
        -------
        DataFrame with columns: dmu, period, EC, TC, ML
        """
        results = []

        for t_idx in range(len(self.years) - 1):
            t, t1 = self.years[t_idx], self.years[t_idx + 1]
            X_t, Y_t, Z_t = self._get_period_data(t)
            X_t1, Y_t1, Z_t1 = self._get_period_data(t1)

            for dmu in self.dmu_ids:
                x_t, y_t, z_t = self._get_dmu_data(dmu, t)
                x_t1, y_t1, z_t1 = self._get_dmu_data(dmu, t1)

                g_x_t, g_y_t, g_z_t = self._get_direction(x_t, y_t, z_t)
                g_x_t1, g_y_t1, g_z_t1 = self._get_direction(x_t1, y_t1, z_t1)

                # Four distance functions
                # D_t(t, t) — period t DMU against period t frontier
                d_tt = self._ddf_distance(x_t, y_t, z_t, X_t, Y_t, Z_t, g_x_t, g_y_t, g_z_t)
                # D_t1(t1, t1)
                d_t1t1 = self._ddf_distance(x_t1, y_t1, z_t1, X_t1, Y_t1, Z_t1, g_x_t1, g_y_t1, g_z_t1)
                # D_t(t1, t1) — period t+1 DMU against period t frontier
                d_tt1 = self._ddf_distance(x_t1, y_t1, z_t1, X_t, Y_t, Z_t, g_x_t1, g_y_t1, g_z_t1)
                # D_t1(t, t) — period t DMU against period t+1 frontier
                d_t1t = self._ddf_distance(x_t, y_t, z_t, X_t1, Y_t1, Z_t1, g_x_t, g_y_t, g_z_t)

                # ML index (Chung et al., 1997)
                num1 = (1 + d_tt)
                den1 = (1 + d_t1t)
                num2 = (1 + d_tt1)
                den2 = (1 + d_t1t1)

                if den1 > 0 and den2 > 0 and num1 > 0 and num2 > 0:
                    ML = np.sqrt((num1 / den1) * (num2 / den2))
                    EC = (1 + d_tt) / (1 + d_t1t1)
                    TC = ML / EC if EC != 0 else np.nan
                else:
                    ML, EC, TC = np.nan, np.nan, np.nan

                results.append({
                    "dmu": dmu,
                    "period": f"{t}-{t1}",
                    "EC": EC,
                    "TC": TC,
                    "ML": ML,
                })

        return pd.DataFrame(results)
