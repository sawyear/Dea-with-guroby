"""Malmquist and Malmquist-Luenberger productivity indices.

References
----------
Fare, R., Grosskopf, S., Norris, M. & Zhang, Z. (1994). Productivity growth,
    technical progress, and efficiency change in industrialized countries.
    American Economic Review, 84(1), 66-83.
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


class Malmquist:
    """Malmquist TFP Index based on radial distance functions.

    Decomposes TFP change into:
    - EC (Efficiency Change) = catch-up effect
    - TC (Technical Change) = frontier shift

    MI = EC * TC

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs : list[str]
    data : pd.DataFrame
        Panel data with 'dmu' and 'year' columns.

    Example
    -------
    >>> mi = Malmquist(inputs=["x1"], desirable_outputs=["y1"],
    ...                undesirable_outputs=["b1"], data=panel_df)
    >>> results = mi.compute(orientation="input")
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
    ) -> None:
        self.input_names = list(inputs)
        self.desirable_names = list(desirable_outputs)
        self.undesirable_names = list(undesirable_outputs)
        self.data = data
        self.dmu_ids = sorted(data["dmu"].unique())
        self.years = sorted(data["year"].unique())
        self.n_dmu = len(self.dmu_ids)
        self._indexed = data.set_index(["dmu", "year"])

    def _get_period_data(self, year):
        """Extract X, Y, Z matrices for a given year."""
        mask = self.data["year"] == year
        d = self.data[mask].sort_values("dmu")
        X = d[self.input_names].to_numpy(dtype=float)
        Y = d[self.desirable_names].to_numpy(dtype=float)
        Z = d[self.undesirable_names].to_numpy(dtype=float)
        return X, Y, Z

    def _get_dmu_data(self, dmu, year):
        row = self._indexed.loc[(dmu, year)]
        x = row[self.input_names].to_numpy(dtype=float)
        y = row[self.desirable_names].to_numpy(dtype=float)
        z = row[self.undesirable_names].to_numpy(dtype=float)
        return x, y, z

    def _radial_distance(
        self,
        x_k, y_k, z_k,
        X_ref, Y_ref, Z_ref,
        orientation: str = "input",
    ) -> float:
        """Solve radial distance of (x_k, y_k, z_k) against reference set."""
        n_ref = X_ref.shape[0]
        m = X_ref.shape[1]
        s1 = Y_ref.shape[1]
        s2 = Z_ref.shape[1]

        mdl = gp.Model()
        mdl.setParam("OutputFlag", 0)
        theta = mdl.addVar(name="theta")
        lam = mdl.addVars(n_ref, name="lambda")
        mdl.update()

        if orientation == "input":
            mdl.setObjective(theta, GRB.MINIMIZE)
            for j in range(m):
                mdl.addConstr(
                    quicksum(X_ref[i, j] * lam[i] for i in range(n_ref))
                    <= theta * x_k[j]
                )
            for j in range(s1):
                mdl.addConstr(
                    quicksum(Y_ref[i, j] * lam[i] for i in range(n_ref))
                    >= y_k[j]
                )
            for j in range(s2):
                mdl.addConstr(
                    quicksum(Z_ref[i, j] * lam[i] for i in range(n_ref))
                    <= theta * z_k[j]
                )
        else:
            mdl.setObjective(theta, GRB.MAXIMIZE)
            for j in range(m):
                mdl.addConstr(
                    quicksum(X_ref[i, j] * lam[i] for i in range(n_ref))
                    <= x_k[j]
                )
            for j in range(s1):
                mdl.addConstr(
                    quicksum(Y_ref[i, j] * lam[i] for i in range(n_ref))
                    >= theta * y_k[j]
                )
            for j in range(s2):
                mdl.addConstr(
                    quicksum(Z_ref[i, j] * lam[i] for i in range(n_ref))
                    <= z_k[j]
                )

        mdl.optimize()
        return mdl.objVal if mdl.status == GRB.OPTIMAL else np.nan

    def compute(
        self, orientation: Literal["input", "output"] = "input"
    ) -> pd.DataFrame:
        """Compute Malmquist index for consecutive year pairs.

        Returns
        -------
        DataFrame with columns: dmu, period, EC, TC, MI
        """
        results = []

        for t_idx in range(len(self.years) - 1):
            t, t1 = self.years[t_idx], self.years[t_idx + 1]
            X_t, Y_t, Z_t = self._get_period_data(t)
            X_t1, Y_t1, Z_t1 = self._get_period_data(t1)

            for k_idx, dmu in enumerate(self.dmu_ids):
                x_t, y_t, z_t = self._get_dmu_data(dmu, t)
                x_t1, y_t1, z_t1 = self._get_dmu_data(dmu, t1)

                # D_t(x_t, y_t) — own period
                d_tt = self._radial_distance(x_t, y_t, z_t, X_t, Y_t, Z_t, orientation)
                # D_t1(x_t1, y_t1) — own period
                d_t1t1 = self._radial_distance(x_t1, y_t1, z_t1, X_t1, Y_t1, Z_t1, orientation)
                # D_t(x_t1, y_t1) — cross period
                d_tt1 = self._radial_distance(x_t1, y_t1, z_t1, X_t, Y_t, Z_t, orientation)
                # D_t1(x_t, y_t) — cross period
                d_t1t = self._radial_distance(x_t, y_t, z_t, X_t1, Y_t1, Z_t1, orientation)

                # Decomposition
                EC = d_t1t1 / d_tt if d_tt != 0 else np.nan
                TC = np.sqrt((d_tt / d_t1t) * (d_tt1 / d_t1t1)) if (d_t1t * d_t1t1) != 0 else np.nan
                MI = EC * TC if not (np.isnan(EC) or np.isnan(TC)) else np.nan

                if orientation == "input":
                    # For input orientation, invert so MI > 1 = improvement
                    EC = 1 / EC if EC != 0 else np.nan
                    MI = 1 / MI if MI != 0 else np.nan

                results.append({
                    "dmu": dmu,
                    "period": f"{t}-{t1}",
                    "EC": EC,
                    "TC": TC,
                    "MI": MI,
                })

        return pd.DataFrame(results)

