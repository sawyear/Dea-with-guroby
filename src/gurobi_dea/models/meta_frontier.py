"""Meta-frontier DEA model.

Computes group-specific frontiers and a global meta-frontier,
then derives the Technology Gap Ratio (TGR).

References
----------
Battese, G.E., Rao, D.S.P. & O'Donnell, C.J. (2004). A metafrontier
    production function for estimation of technical efficiencies and
    technology gaps for firms operating under different technologies.
    Journal of Productivity Analysis, 21(1), 91-103.
O'Donnell, C.J., Rao, D.S.P. & Battese, G.E. (2008). Metafrontier
    frameworks for the study of firm-level efficiencies and technology
    ratios. Empirical Economics, 34(2), 231-255.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase


class MetaFrontier(DEABase):
    """Meta-frontier DEA with Technology Gap Ratio.

    Computes:
    - TE_group: efficiency relative to group-specific frontier
    - TE_meta: efficiency relative to the global meta-frontier
    - TGR = TE_meta / TE_group (Technology Gap Ratio)

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs, data, dmu_col :
        Inherited from DEABase.
    group_col : str
        Column name identifying group membership (e.g., region, industry).
    method : str
        Underlying model for distance computation:
        ``'radial'`` (default) or ``'sbm'``.

    Example
    -------
    >>> model = MetaFrontier(
    ...     inputs=["x1"], desirable_outputs=["y1"],
    ...     undesirable_outputs=["b1"], data=df,
    ...     group_col="region",
    ... )
    >>> results = model.solve(scale="c")
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        group_col: str,
        dmu_col: str = "dmu",
        method: str = "radial",
    ) -> None:
        super().__init__(inputs, desirable_outputs, undesirable_outputs, data, dmu_col)
        self.group_col = group_col
        self.method = method
        self.groups = data[group_col].unique()

    def _radial_distance(
        self, k: int, X_ref: np.ndarray, Y_ref: np.ndarray, Z_ref: np.ndarray,
        scale: str,
    ) -> float:
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
                <= theta * self.X[k, j]
            )
        for j in range(self.s1):
            mdl.addConstr(
                quicksum(Y_ref[i, j] * lam[i] for i in range(n_ref))
                >= self.Y[k, j]
            )
        for j in range(self.s2):
            mdl.addConstr(
                quicksum(Z_ref[i, j] * lam[i] for i in range(n_ref))
                <= theta * self.Z[k, j]
            )
        if scale == "v":
            mdl.addConstr(quicksum(lam[i] for i in range(n_ref)) == 1)

        mdl.optimize()
        return mdl.objVal if mdl.status == GRB.OPTIMAL else np.nan

    def _sbm_distance(
        self, k: int, X_ref: np.ndarray, Y_ref: np.ndarray, Z_ref: np.ndarray,
        scale: str,
    ) -> float:
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
            t - (1 / self.m) * quicksum(s_neg[i] / self.X[k, i] for i in range(self.m)),
            GRB.MINIMIZE,
        )
        for j in range(self.m):
            mdl.addConstr(
                quicksum(X_ref[i, j] * lam[i] for i in range(n_ref))
                == t * self.X[k, j] - s_neg[j]
            )
        for j in range(self.s1):
            mdl.addConstr(
                quicksum(Y_ref[i, j] * lam[i] for i in range(n_ref))
                == t * self.Y[k, j] + s_pos[j]
            )
        for j in range(self.s2):
            mdl.addConstr(
                quicksum(Z_ref[i, j] * lam[i] for i in range(n_ref))
                == t * self.Z[k, j] - s_und[j]
            )
        denom = self.s1 + self.s2
        mdl.addConstr(
            t + (1 / denom) * (
                quicksum(s_pos[i] / self.Y[k, i] for i in range(self.s1))
                + quicksum(s_und[i] / self.Z[k, i] for i in range(self.s2))
            ) == 1
        )
        if scale == "v":
            mdl.addConstr(quicksum(lam[i] for i in range(n_ref)) == t)

        mdl.optimize()
        return mdl.objVal if mdl.status == GRB.OPTIMAL else np.nan

    def _compute_distance(self, k, X_ref, Y_ref, Z_ref, scale):
        if self.method == "sbm":
            return self._sbm_distance(k, X_ref, Y_ref, Z_ref, scale)
        return self._radial_distance(k, X_ref, Y_ref, Z_ref, scale)

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        res = self._empty_results()
        res["group"] = self.data[self.group_col].values
        res["TE_group"] = np.nan
        res["TE_meta"] = np.nan
        res["TGR"] = np.nan

        # Step 1: group-specific frontiers
        for grp in self.groups:
            grp_mask = self.data[self.group_col] == grp
            grp_idx = np.where(grp_mask)[0]
            X_grp = self.X[grp_idx]
            Y_grp = self.Y[grp_idx]
            Z_grp = self.Z[grp_idx]

            for k in grp_idx:
                te = self._compute_distance(k, X_grp, Y_grp, Z_grp, scale)
                res.at[k, "TE_group"] = te

        # Step 2: meta-frontier (all DMUs pooled)
        for k in range(self.n_dmu):
            te = self._compute_distance(k, self.X, self.Y, self.Z, scale)
            res.at[k, "TE_meta"] = te

        # Step 3: TGR
        res["TGR"] = res["TE_meta"] / res["TE_group"]
        res["TE"] = res["TE_meta"]

        return res
