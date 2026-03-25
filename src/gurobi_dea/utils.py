"""Utility functions for DEA models.

Includes affinity-matrix computation (Tone & Tsutsui, 2010) used by EBM.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh


def s_corr(a: NDArray, b: NDArray) -> float:
    """Compute the S-correlation between two slack vectors.

    Parameters
    ----------
    a, b : 1-D arrays of the same length (one per DMU).

    Returns
    -------
    float in [0, 1].  1 means perfectly correlated slacks.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.log(np.asarray(b, dtype=float) / np.asarray(a, dtype=float))
    rng = float(np.nanmax(c) - np.nanmin(c))
    if rng == 0:
        return 1.0
    d = float(np.nanmean(np.abs(c - np.nanmean(c))))
    return 1.0 - 2.0 * d / rng


def affinity_matrix(slack_matrix: NDArray) -> tuple[float, NDArray]:
    """Compute epsilon and weight vector from a slack matrix.

    Parameters
    ----------
    slack_matrix : 2-D array, shape (n_dmu, n_vars).

    Returns
    -------
    epsilon : float – mixing parameter in [0, 1].
    weights : 1-D array of length n_vars, sums to 1.
    """
    n_vars = slack_matrix.shape[1]
    if n_vars <= 1:
        return 0.0, np.ones(1)

    S = np.eye(n_vars)
    for i in range(n_vars):
        for j in range(n_vars):
            S[i, j] = s_corr(slack_matrix[:, i], slack_matrix[:, j])

    eigval, eigvec = eigh(S, subset_by_index=[n_vars - 1, n_vars - 1])
    rho = float(eigval[0])
    w = eigvec[:, 0]
    epsilon = (n_vars - rho) / (n_vars - 1)
    weights = np.abs(w) / np.abs(w).sum()  # ensure positive & normalized
    return float(epsilon), weights
