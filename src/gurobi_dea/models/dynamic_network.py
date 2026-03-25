"""Dynamic Network SBM-DEA.

Combines network structure (multiple divisions connected by links)
with dynamic structure (carry-over activities between periods).

References
----------
Tone, K. & Tsutsui, M. (2014). Dynamic DEA with network structure:
    A slacks-based measure approach. Omega, 42(1), 124-131.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum


class DynamicNetworkSBM:
    """Dynamic Network SBM-DEA.

    Handles both within-period network links (between divisions)
    and between-period carry-overs simultaneously.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with 'dmu' and 'year' columns.
    stages : list[dict]
        Each dict defines one division/stage per period:
        ``'x'`` (inputs), ``'y'`` (desirable outputs), ``'z'`` (undesirable outputs).
    links : list[dict]
        Within-period links between stages. Each dict:
        ``'vars'`` (column names), ``'from'`` (stage index), ``'to'`` (stage index).
    carry_overs : list[dict]
        Between-period carry-overs. Each dict:
        ``'vars'`` (column names), ``'stage'`` (which stage it belongs to).
    weight_year : list[float] | None
        Year weights (default: equal).

    Example
    -------
    >>> model = DynamicNetworkSBM(
    ...     data=panel_df,
    ...     stages=[
    ...         {"x": ["x1"], "y": ["y1"], "z": ["z1"]},
    ...         {"x": ["x2"], "y": ["y2"], "z": ["z2"]},
    ...     ],
    ...     links=[{"vars": ["w12"], "from": 0, "to": 1}],
    ...     carry_overs=[{"vars": ["k1"], "stage": 0}],
    ... )
    >>> results = model.solve(scale="c")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        stages: list[dict],
        links: list[dict] | None = None,
        carry_overs: list[dict] | None = None,
        weight_year: list[float] | None = None,
    ) -> None:
        self.data = data
        self.stages = stages
        self.links = links or []
        self.carry_overs = carry_overs or []
        self.n_stages = len(stages)

        self.dmu_ids = sorted(data["dmu"].unique())
        self.years = sorted(data["year"].unique())
        self.n_dmu = len(self.dmu_ids)
        self.n_year = len(self.years)

        self.w_year = (
            np.array(weight_year) if weight_year
            else np.ones(self.n_year) / self.n_year
        )
        self._indexed = data.set_index(["dmu", "year"])

    def _get_stage_data(self, year, stage_idx):
        """Get (n_dmu, n_vars) matrix for a stage in a given year."""
        result = {}
        stg = self.stages[stage_idx]
        for key in ("x", "y", "z"):
            cols = stg.get(key, [])
            if cols:
                arr = np.array([
                    self._indexed.loc[(d, year), cols].to_numpy(dtype=float)
                    for d in self.dmu_ids
                ])
                result[key] = arr
            else:
                result[key] = np.empty((self.n_dmu, 0))
        return result

    def _get_link_data(self, year, link_idx):
        cols = self.links[link_idx]["vars"]
        return np.array([
            self._indexed.loc[(d, year), cols].to_numpy(dtype=float)
            for d in self.dmu_ids
        ])

    def _get_carry_data(self, year, co_idx):
        cols = self.carry_overs[co_idx]["vars"]
        return np.array([
            self._indexed.loc[(d, year), cols].to_numpy(dtype=float)
            for d in self.dmu_ids
        ])

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = pd.DataFrame(
            columns=["dmu", "TE"] + [str(y) for y in self.years],
            index=range(self.n_dmu),
        )
        res["dmu"] = self.dmu_ids

        for k_idx in range(self.n_dmu):
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)
            mdl.setParam("NonConvex", 2)

            t = mdl.addVar(name="t")
            x_k = mdl.addVars(self.n_year, name="x_k")  # input-side per year
            y_k = mdl.addVars(self.n_year, name="y_k")  # output-side per year

            # Per (stage, year) variables
            s_x, s_y, s_z, lam = {}, {}, {}, {}
            s_co = {}  # carry-over slacks

            for yr_idx in range(self.n_year):
                for p in range(self.n_stages):
                    sd = self._get_stage_data(self.years[yr_idx], p)
                    nx = sd["x"].shape[1]
                    ny = sd["y"].shape[1]
                    nz = sd["z"].shape[1]
                    s_x[p, yr_idx] = mdl.addVars(nx, name=f"sx_{p}_{yr_idx}")
                    s_y[p, yr_idx] = mdl.addVars(ny, name=f"sy_{p}_{yr_idx}")
                    s_z[p, yr_idx] = mdl.addVars(nz, name=f"sz_{p}_{yr_idx}")
                    lam[p, yr_idx] = mdl.addVars(self.n_dmu, name=f"lam_{p}_{yr_idx}")

                for co_idx in range(len(self.carry_overs)):
                    co_data = self._get_carry_data(self.years[yr_idx], co_idx)
                    n_co = co_data.shape[1]
                    s_co[co_idx, yr_idx] = mdl.addVars(n_co, name=f"sco_{co_idx}_{yr_idx}")

            mdl.update()

            # Build constraints per year
            total_input_vars = 0
            total_output_vars = 0
            total_carry_vars = 0
            for p in range(self.n_stages):
                total_input_vars += len(self.stages[p].get("x", []))
                total_output_vars += len(self.stages[p].get("y", [])) + len(self.stages[p].get("z", []))
            for co in self.carry_overs:
                total_carry_vars += len(co.get("vars", []))

            for yr_idx, yr in enumerate(self.years):
                # Stage constraints
                input_obj_parts = []
                output_norm_parts = []

                for p in range(self.n_stages):
                    sd = self._get_stage_data(yr, p)
                    nx, ny, nz = sd["x"].shape[1], sd["y"].shape[1], sd["z"].shape[1]
                    x_k0 = sd["x"][k_idx]
                    y_k0 = sd["y"][k_idx]
                    z_k0 = sd["z"][k_idx]

                    # Input constraints
                    for j in range(nx):
                        mdl.addConstr(
                            quicksum(sd["x"][i, j] * lam[p, yr_idx][i] for i in range(self.n_dmu))
                            == t * x_k0[j] - s_x[p, yr_idx][j]
                        )
                        if x_k0[j] > 0:
                            input_obj_parts.append(s_x[p, yr_idx][j] / x_k0[j])

                    # Desirable output constraints
                    for j in range(ny):
                        mdl.addConstr(
                            quicksum(sd["y"][i, j] * lam[p, yr_idx][i] for i in range(self.n_dmu))
                            == t * y_k0[j] + s_y[p, yr_idx][j]
                        )
                        if y_k0[j] > 0:
                            output_norm_parts.append(s_y[p, yr_idx][j] / y_k0[j])

                    # Undesirable output constraints
                    for j in range(nz):
                        mdl.addConstr(
                            quicksum(sd["z"][i, j] * lam[p, yr_idx][i] for i in range(self.n_dmu))
                            == t * z_k0[j] - s_z[p, yr_idx][j]
                        )
                        if z_k0[j] > 0:
                            output_norm_parts.append(s_z[p, yr_idx][j] / z_k0[j])

                    # VRS per stage
                    if scale == "v":
                        mdl.addConstr(
                            quicksum(lam[p, yr_idx][i] for i in range(self.n_dmu)) == t
                        )

                # Carry-over constraints
                carry_obj_parts = []
                for co_idx, co in enumerate(self.carry_overs):
                    co_data = self._get_carry_data(yr, co_idx)
                    p_stage = co.get("stage", 0)
                    n_co = co_data.shape[1]
                    co_k0 = co_data[k_idx]

                    for j in range(n_co):
                        mdl.addConstr(
                            quicksum(co_data[i, j] * lam[p_stage, yr_idx][i] for i in range(self.n_dmu))
                            == t * co_k0[j] - s_co[co_idx, yr_idx][j]
                        )
                        if co_k0[j] > 0:
                            carry_obj_parts.append(s_co[co_idx, yr_idx][j] / co_k0[j])

                    # Carry-over link between periods
                    if yr_idx + 1 < self.n_year:
                        co_next = self._get_carry_data(self.years[yr_idx + 1], co_idx)
                        for j in range(n_co):
                            mdl.addConstr(
                                quicksum(co_data[i, j] * lam[p_stage, yr_idx][i] for i in range(self.n_dmu))
                                == quicksum(co_next[i, j] * lam[p_stage, yr_idx + 1][i] for i in range(self.n_dmu))
                            )

                # Within-period link constraints
                for lk_idx, lk in enumerate(self.links):
                    lk_data = self._get_link_data(yr, lk_idx)
                    p_from, p_to = lk["from"], lk["to"]
                    for j in range(lk_data.shape[1]):
                        mdl.addConstr(
                            quicksum(lk_data[i, j] * lam[p_from, yr_idx][i] for i in range(self.n_dmu))
                            == quicksum(lk_data[i, j] * lam[p_to, yr_idx][i] for i in range(self.n_dmu))
                        )

                # Per-year efficiency components
                denom_in = total_input_vars + total_carry_vars
                denom_out = total_output_vars
                if denom_in > 0:
                    mdl.addConstr(
                        x_k[yr_idx] == t - (1.0 / denom_in) * quicksum(input_obj_parts + carry_obj_parts)
                    )
                else:
                    mdl.addConstr(x_k[yr_idx] == t)

                if denom_out > 0:
                    mdl.addConstr(
                        y_k[yr_idx] == t + (1.0 / denom_out) * quicksum(output_norm_parts)
                    )
                else:
                    mdl.addConstr(y_k[yr_idx] == t)

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

            res.at[k_idx, "TE"] = mdl.objVal if mdl.status == GRB.OPTIMAL else np.nan
            for yr_idx in range(self.n_year):
                val = x_k[yr_idx].X if mdl.status == GRB.OPTIMAL else np.nan
                res.loc[k_idx, str(self.years[yr_idx])] = val

        return res
