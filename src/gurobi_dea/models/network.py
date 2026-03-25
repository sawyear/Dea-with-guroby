"""Network DEA models (SBM and EBM variants).

Supports up to 3-stage network structures with intermediate links,
undesirable outputs at each stage.

References
----------
Tone, K. & Tsutsui, M. (2009). Network DEA: A slacks-based measure approach.
    European Journal of Operational Research, 197(1), 243-252.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase
from gurobi_dea.utils import affinity_matrix


class NetworkSBM(DEABase):
    """Network SBM-DEA with undesirable outputs (up to 3 stages).

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs, data, dmu_col :
        Inherited from DEABase (used for data validation only).
    stages : list[dict]
        Each dict defines one stage with keys:
        ``'x'`` (inputs), ``'y'`` (desirable outputs), ``'z'`` (undesirable outputs).
    links : list[dict]
        Each dict has ``'vars'`` (list of column names) linking stage i -> i+1.
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        stages: list[dict],
        links: list[dict],
        dmu_col: str = "dmu",
    ) -> None:
        super().__init__(inputs, desirable_outputs, undesirable_outputs, data, dmu_col)
        self.stages = stages
        self.links = links
        self.n_stages = len(stages)

        # Pre-extract numpy arrays per stage
        self._stage_data: list[dict[str, np.ndarray]] = []
        for s in stages:
            self._stage_data.append({
                "X": data[s["x"]].to_numpy(dtype=float) if s.get("x") else np.empty((self.n_dmu, 0)),
                "Y": data[s["y"]].to_numpy(dtype=float) if s.get("y") else np.empty((self.n_dmu, 0)),
                "Z": data[s["z"]].to_numpy(dtype=float) if s.get("z") else np.empty((self.n_dmu, 0)),
            })
        self._link_data: list[np.ndarray] = []
        for lk in links:
            self._link_data.append(data[lk["vars"]].to_numpy(dtype=float))


    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = self._empty_results()
        T = self.n_stages

        for k in range(self.n_dmu):
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)
            mdl.setParam("NonConvex", 2)

            t = mdl.addVar(name="t")  # C-C scalar

            # Per-stage variables
            s_x, s_y, s_z, lam = {}, {}, {}, {}
            for p in range(T):
                sd = self._stage_data[p]
                nx, ny, nz = sd["X"].shape[1], sd["Y"].shape[1], sd["Z"].shape[1]
                s_x[p] = mdl.addVars(nx, name=f"sx{p}")
                s_y[p] = mdl.addVars(ny, name=f"sy{p}")
                s_z[p] = mdl.addVars(nz, name=f"sz{p}")
                lam[p] = mdl.addVars(self.n_dmu, name=f"lam{p}")
            mdl.update()

            # --- Objective: minimize input-side ---
            obj_parts = []
            for p in range(T):
                sd = self._stage_data[p]
                nx = sd["X"].shape[1]
                if nx > 0:
                    obj_parts.append(
                        quicksum(s_x[p][j] / (nx * sd["X"][k, j]) for j in range(nx))
                    )
            mdl.setObjective(t - (1 / T) * quicksum(obj_parts), GRB.MINIMIZE)

            # --- Constraints per stage ---
            for p in range(T):
                sd = self._stage_data[p]
                nx, ny, nz = sd["X"].shape[1], sd["Y"].shape[1], sd["Z"].shape[1]
                for j in range(nx):
                    mdl.addConstr(
                        quicksum(sd["X"][i, j] * lam[p][i] for i in range(self.n_dmu))
                        == t * sd["X"][k, j] - s_x[p][j]
                    )
                for j in range(ny):
                    mdl.addConstr(
                        quicksum(sd["Y"][i, j] * lam[p][i] for i in range(self.n_dmu))
                        == t * sd["Y"][k, j] + s_y[p][j]
                    )
                for j in range(nz):
                    mdl.addConstr(
                        quicksum(sd["Z"][i, j] * lam[p][i] for i in range(self.n_dmu))
                        == t * sd["Z"][k, j] - s_z[p][j]
                    )
                if scale == "v":
                    mdl.addConstr(quicksum(lam[p][i] for i in range(self.n_dmu)) == t)

            # --- Link constraints ---
            for idx, lk_arr in enumerate(self._link_data):
                n_link = lk_arr.shape[1]
                p_from, p_to = idx, idx + 1
                for j in range(n_link):
                    mdl.addConstr(
                        quicksum(lk_arr[i, j] * lam[p_from][i] for i in range(self.n_dmu))
                        == quicksum(lk_arr[i, j] * lam[p_to][i] for i in range(self.n_dmu))
                    )

            # --- Normalization (output side == 1) ---
            norm_parts = []
            for p in range(T):
                sd = self._stage_data[p]
                ny, nz = sd["Y"].shape[1], sd["Z"].shape[1]
                denom = ny + nz
                if denom > 0:
                    norm_parts.append(
                        quicksum(s_y[p][j] / (denom * sd["Y"][k, j]) for j in range(ny))
                        + quicksum(s_z[p][j] / (denom * sd["Z"][k, j]) for j in range(nz))
                    )
            mdl.addConstr(t + (1 / T) * quicksum(norm_parts) == 1)

            mdl.optimize()

            # Store results
            t_val = t.X if t.X != 0 else 1.0
            res.at[k, "TE"] = mdl.objVal
            for p in range(T):
                sd = self._stage_data[p]
                for j, col in enumerate(self.stages[p].get("x", [])):
                    res.loc[k, col] = s_x[p][j].X / t_val
                for j, col in enumerate(self.stages[p].get("y", [])):
                    res.loc[k, col] = s_y[p][j].X / t_val
                for j, col in enumerate(self.stages[p].get("z", [])):
                    res.loc[k, col] = s_z[p][j].X / t_val

        return res


class NetworkEBM(NetworkSBM):
    """Network EBM-DEA with undesirable outputs (up to 3 stages).

    Extends NetworkSBM by computing affinity-matrix weights per stage
    and using the epsilon-based objective.

    Note: This is a simplified implementation. For production use with
    many stages, consider subclassing and overriding _compute_stage_params.
    """

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        # Step 1: run SBM(VRS) per stage to get slacks for affinity matrix
        from gurobi_dea.models.sbm import SBM

        stage_params: list[dict] = []
        for p, stg in enumerate(self.stages):
            x_cols = stg.get("x", [])
            y_cols = stg.get("y", [])
            z_cols = stg.get("z", [])
            if not x_cols or not y_cols or not z_cols:
                stage_params.append({"eps_x": 0, "w_x": np.ones(1),
                                     "eps_y": 0, "w_y": np.ones(1),
                                     "eps_z": 0, "w_z": np.ones(1)})
                continue
            sbm = SBM(inputs=x_cols, desirable_outputs=y_cols,
                       undesirable_outputs=z_cols, data=self.data, dmu_col=self.dmu_col)
            sbm_res = sbm.solve(scale="v")
            sd = self._stage_data[p]
            eps_x, w_x = affinity_matrix(sd["X"] - sbm_res[x_cols].to_numpy(dtype=float))
            eps_y, w_y = affinity_matrix(sbm_res[y_cols].to_numpy(dtype=float) + sd["Y"])
            eps_z, w_z = affinity_matrix(sd["Z"] - sbm_res[z_cols].to_numpy(dtype=float))
            stage_params.append({"eps_x": eps_x, "w_x": w_x,
                                 "eps_y": eps_y, "w_y": w_y,
                                 "eps_z": eps_z, "w_z": w_z})

        # Step 2: solve EBM per DMU
        res = self._empty_results()
        T = self.n_stages

        for k in range(self.n_dmu):
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)
            mdl.setParam("NonConvex", 2)

            theta = mdl.addVars(T, name="theta")
            eta = mdl.addVars(T, name="eta")
            s_x, s_y, s_z, lam = {}, {}, {}, {}
            for p in range(T):
                sd = self._stage_data[p]
                nx, ny, nz = sd["X"].shape[1], sd["Y"].shape[1], sd["Z"].shape[1]
                s_x[p] = mdl.addVars(nx, name=f"sx{p}")
                s_y[p] = mdl.addVars(ny, name=f"sy{p}")
                s_z[p] = mdl.addVars(nz, name=f"sz{p}")
                lam[p] = mdl.addVars(self.n_dmu, name=f"lam{p}")
            mdl.update()

            # Objective
            obj_parts = []
            for p in range(T):
                sd = self._stage_data[p]
                sp = stage_params[p]
                nx = sd["X"].shape[1]
                obj_parts.append(theta[p])
                if nx > 0:
                    obj_parts.append(
                        -sp["eps_x"] * quicksum(
                            float(sp["w_x"][j]) * s_x[p][j] / sd["X"][k, j]
                            for j in range(nx)
                        )
                    )
            mdl.setObjective(quicksum(obj_parts), GRB.MINIMIZE)

            # Constraints per stage
            for p in range(T):
                sd = self._stage_data[p]
                nx, ny, nz = sd["X"].shape[1], sd["Y"].shape[1], sd["Z"].shape[1]
                for j in range(nx):
                    mdl.addConstr(
                        quicksum(sd["X"][i, j] * lam[p][i] for i in range(self.n_dmu))
                        == theta[p] * sd["X"][k, j] - s_x[p][j]
                    )
                for j in range(ny):
                    mdl.addConstr(
                        quicksum(sd["Y"][i, j] * lam[p][i] for i in range(self.n_dmu))
                        == eta[p] * sd["Y"][k, j] + s_y[p][j]
                    )
                for j in range(nz):
                    mdl.addConstr(
                        quicksum(sd["Z"][i, j] * lam[p][i] for i in range(self.n_dmu))
                        == eta[p] * sd["Z"][k, j] - s_z[p][j]
                    )
                if scale == "v":
                    mdl.addConstr(quicksum(lam[p][i] for i in range(self.n_dmu)) == 1)

            # Link constraints
            for idx, lk_arr in enumerate(self._link_data):
                n_link = lk_arr.shape[1]
                p_from, p_to = idx, idx + 1
                for j in range(n_link):
                    mdl.addConstr(
                        quicksum(lk_arr[i, j] * lam[p_from][i] for i in range(self.n_dmu))
                        == quicksum(lk_arr[i, j] * lam[p_to][i] for i in range(self.n_dmu))
                    )

            # Normalization (output side)
            norm_parts = []
            for p in range(T):
                sd = self._stage_data[p]
                sp = stage_params[p]
                ny, nz = sd["Y"].shape[1], sd["Z"].shape[1]
                norm_parts.append(eta[p])
                if ny > 0:
                    norm_parts.append(
                        sp["eps_y"] * quicksum(
                            float(sp["w_y"][j]) * s_y[p][j] / sd["Y"][k, j]
                            for j in range(ny)
                        )
                    )
                if nz > 0:
                    norm_parts.append(
                        sp["eps_z"] * quicksum(
                            float(sp["w_z"][j]) * s_z[p][j] / sd["Z"][k, j]
                            for j in range(nz)
                        )
                    )
            mdl.addConstr(quicksum(norm_parts) == 1)

            mdl.optimize()

            res.at[k, "TE"] = mdl.objVal
            for p in range(T):
                for j, col in enumerate(self.stages[p].get("x", [])):
                    res.loc[k, col] = s_x[p][j].X
                for j, col in enumerate(self.stages[p].get("y", [])):
                    res.loc[k, col] = s_y[p][j].X
                for j, col in enumerate(self.stages[p].get("z", [])):
                    res.loc[k, col] = s_z[p][j].X

        return res
