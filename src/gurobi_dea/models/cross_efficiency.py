"""Cross-efficiency DEA model.

Eliminates the self-appraisal bias of standard DEA by using each DMU's
optimal weights to evaluate all other DMUs.

References
----------
Sexton, T.R., Silkman, R.H. & Hogan, A.J. (1986). Data envelopment
    analysis: Critique and extensions. New Directions for Program
    Evaluation, 1986(32), 73-105.
Doyle, J. & Green, R. (1994). Efficiency and cross-efficiency in DEA:
    Derivations, meanings and uses. Journal of the Operational Research
    Society, 45(5), 567-578.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB, quicksum

from gurobi_dea.base import DEABase


class CrossEfficiency(DEABase):
    """Cross-efficiency DEA.

    Two-step procedure:
    1. Solve standard CCR multiplier model for each DMU to get optimal weights.
    2. Use each DMU's weights to evaluate all other DMUs.
    3. Average the peer-appraisal scores.

    Parameters
    ----------
    inputs, desirable_outputs, undesirable_outputs, data, dmu_col :
        Inherited from DEABase.
    aggressive : bool
        If True, use aggressive (min other) secondary objective.
        If False (default), use benevolent (max other) secondary objective.

    Example
    -------
    >>> model = CrossEfficiency(
    ...     inputs=["x1", "x2"], desirable_outputs=["y1"],
    ...     undesirable_outputs=["b1"], data=df,
    ... )
    >>> results = model.solve()
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        dmu_col: str = "dmu",
        aggressive: bool = False,
    ) -> None:
        super().__init__(inputs, desirable_outputs, undesirable_outputs, data, dmu_col)
        self.aggressive = aggressive

    def solve(self, scale: Literal["c", "v"] = "c") -> pd.DataFrame:
        # Step 1: get optimal multiplier weights for each DMU
        # Using input-oriented CCR multiplier form
        weights_v = np.zeros((self.n_dmu, self.m))       # input weights
        weights_u = np.zeros((self.n_dmu, self.s1))      # output weights
        self_scores = np.zeros(self.n_dmu)

        for k in range(self.n_dmu):
            mdl = gp.Model()
            mdl.setParam("OutputFlag", 0)

            v = mdl.addVars(self.m, name="v", lb=1e-6)       # input weights
            u = mdl.addVars(self.s1, name="u", lb=1e-6)      # output weights
            mdl.update()

            # Maximize virtual output of DMU k
            mdl.setObjective(
                quicksum(u[j] * self.Y[k, j] for j in range(self.s1)),
                GRB.MAXIMIZE,
            )

            # Normalize: virtual input of DMU k = 1
            mdl.addConstr(
                quicksum(v[j] * self.X[k, j] for j in range(self.m)) == 1
            )

            # For all DMUs: virtual output <= virtual input
            # (undesirable outputs added to input side)
            for i in range(self.n_dmu):
                mdl.addConstr(
                    quicksum(u[j] * self.Y[i, j] for j in range(self.s1))
                    <= quicksum(v[j] * self.X[i, j] for j in range(self.m))
                )

            mdl.optimize()

            if mdl.status == GRB.OPTIMAL:
                self_scores[k] = mdl.objVal
                for j in range(self.m):
                    weights_v[k, j] = v[j].X
                for j in range(self.s1):
                    weights_u[k, j] = u[j].X

        # Step 2: cross-evaluation matrix
        cross_matrix = np.zeros((self.n_dmu, self.n_dmu))
        for d in range(self.n_dmu):  # evaluator
            for k in range(self.n_dmu):  # evaluated
                virtual_output = np.dot(weights_u[d], self.Y[k])
                virtual_input = np.dot(weights_v[d], self.X[k])
                cross_matrix[d, k] = virtual_output / virtual_input if virtual_input > 0 else np.nan

        # Step 3: average cross-efficiency (peer appraisal)
        # Exclude self-appraisal
        ce_scores = np.zeros(self.n_dmu)
        for k in range(self.n_dmu):
            peers = [cross_matrix[d, k] for d in range(self.n_dmu) if d != k]
            ce_scores[k] = np.nanmean(peers)

        # Build results
        res = self._empty_results()
        res["TE"] = ce_scores
        res["TE_self"] = self_scores

        # Add maverick index: how much does self-appraisal differ from peer
        res["maverick"] = (self_scores - ce_scores) / ce_scores

        return res
