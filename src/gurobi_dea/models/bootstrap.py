"""Bootstrap DEA with Simar-Wilson two-stage procedure.

References
----------
Simar, L. & Wilson, P.W. (2007). Estimation and inference in two-stage,
    semi-parametric models of production processes. Journal of Econometrics,
    136(1), 31-64.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase


class BootstrapDEA(DEABase):
    """Bootstrap DEA with bias correction and confidence intervals.

    Implements the Simar-Wilson (2007) Algorithm #1:
    1. Compute original DEA scores.
    2. Bootstrap the scores B times via smoothed resampling.
    3. Compute bias-corrected scores and confidence intervals.

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs, data, dmu_col :
        Inherited from DEABase.
    n_bootstrap : int
        Number of bootstrap replications (default 2000).
    alpha : float
        Significance level for confidence intervals (default 0.05).
    seed : int
        Random seed for reproducibility.

    Example
    -------
    >>> model = BootstrapDEA(
    ...     inputs=["x1"], desirable_outputs=["y1"],
    ...     undesirable_outputs=["b1"], data=df,
    ...     n_bootstrap=2000,
    ... )
    >>> results = model.solve(scale="c")
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        dmu_col: str = "dmu",
        n_bootstrap: int = 2000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__(inputs, desirable_outputs, undesirable_outputs, data, dmu_col)
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def _radial_score(
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

    def _smooth_bootstrap_sample(self, scores: np.ndarray) -> np.ndarray:
        """Generate smoothed bootstrap sample using reflection method."""
        n = len(scores)
        # Silverman bandwidth
        h = 1.06 * np.std(scores) * n ** (-1 / 5)

        # Resample with replacement
        idx = self.rng.choice(n, size=n, replace=True)
        beta_star = scores[idx] + h * self.rng.standard_normal(n)

        # Reflect to ensure scores stay in (0, 1] for input orientation
        beta_star = np.where(beta_star > 1, 2 - beta_star, beta_star)
        beta_star = np.where(beta_star <= 0, np.abs(beta_star) + 1e-6, beta_star)

        return beta_star

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        # Step 1: original DEA scores
        original_scores = np.zeros(self.n_dmu)
        for k in range(self.n_dmu):
            original_scores[k] = self._radial_score(k, self.X, self.Y, self.Z, scale)

        # Step 2: bootstrap
        boot_scores = np.zeros((self.n_bootstrap, self.n_dmu))
        for b in range(self.n_bootstrap):
            # Smoothed bootstrap resample
            beta_star = self._smooth_bootstrap_sample(original_scores)

            # Rescale data: X_star = X * (original / beta_star)
            scale_factor = original_scores / beta_star
            X_star = self.X * scale_factor[:, np.newaxis]
            Y_star = self.Y.copy()
            Z_star = self.Z * scale_factor[:, np.newaxis]

            for k in range(self.n_dmu):
                boot_scores[b, k] = self._radial_score(k, X_star, Y_star, Z_star, scale)

        # Step 3: bias correction and CI
        bias = boot_scores.mean(axis=0) - original_scores
        corrected = original_scores - bias

        ci_lower = np.percentile(boot_scores, 100 * self.alpha / 2, axis=0)
        ci_upper = np.percentile(boot_scores, 100 * (1 - self.alpha / 2), axis=0)

        res = self._empty_results()
        res["TE"] = corrected
        res["TE_original"] = original_scores
        res["bias"] = bias
        res["ci_lower"] = ci_lower
        res["ci_upper"] = ci_upper

        return res
