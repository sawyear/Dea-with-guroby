"""Base class for all DEA models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class DEABase(ABC):
    """Abstract base for static (cross-sectional) DEA models.

    Parameters
    ----------
    inputs : list[str]
        Column names for input variables.
    desirable_outputs : list[str]
        Column names for desirable output variables.
    undesirable_outputs : list[str]
        Column names for undesirable output variables.
    dmu_col : str
        Column that identifies each DMU (default ``'dmu'``).
    data : pd.DataFrame
        The dataset.  Must contain *dmu_col* and all variable columns.
    """

    def __init__(
        self,
        inputs: list[str],
        desirable_outputs: list[str],
        undesirable_outputs: list[str],
        data: pd.DataFrame,
        dmu_col: str = "dmu",
    ) -> None:
        self.input_names = list(inputs)
        self.desirable_names = list(desirable_outputs)
        self.undesirable_names = list(undesirable_outputs)
        self.dmu_col = dmu_col
        self.data = data.reset_index(drop=True)

        # Dimensions
        self.n_dmu = len(self.data)
        self.m = len(self.input_names)       # number of inputs
        self.s1 = len(self.desirable_names)  # number of desirable outputs
        self.s2 = len(self.undesirable_names)  # number of undesirable outputs

        # Numpy matrices for fast access
        self.X: NDArray = self.data[self.input_names].to_numpy(dtype=float)
        self.Y: NDArray = self.data[self.desirable_names].to_numpy(dtype=float)
        self.Z: NDArray = self.data[self.undesirable_names].to_numpy(dtype=float)

        self._validate()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """Basic sanity checks."""
        missing_cols = (
            set(self.input_names + self.desirable_names + self.undesirable_names)
            - set(self.data.columns)
        )
        if missing_cols:
            raise ValueError(f"Columns not found in data: {missing_cols}")
        if (self.X <= 0).any() or (self.Y <= 0).any() or (self.Z <= 0).any():
            import warnings
            warnings.warn(
                "Non-positive values detected in inputs/outputs. "
                "This may cause division-by-zero in slack normalization.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    def _empty_results(self) -> pd.DataFrame:
        """Create an empty results DataFrame."""
        cols = (
            [self.dmu_col, "TE"]
            + self.input_names
            + self.desirable_names
            + self.undesirable_names
        )
        res = pd.DataFrame(np.nan, index=range(self.n_dmu), columns=cols)
        res[self.dmu_col] = self.data[self.dmu_col].values
        return res

    # ------------------------------------------------------------------
    @abstractmethod
    def solve(
        self, scale: Literal["c", "v"] = "c"
    ) -> pd.DataFrame:
        """Solve the model and return a results DataFrame.

        Parameters
        ----------
        scale : 'c' or 'v'
            'c' = CRS (constant returns to scale),
            'v' = VRS (variable returns to scale).
        """
        ...
