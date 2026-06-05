"""Probability calibration (isotonic / Platt).

`IsotonicCalibrator` wraps `sklearn.isotonic.IsotonicRegression` to
transform raw P(Y=1) into a calibrated probability. The calibration
is fit on out-of-fold predictions to avoid leakage.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Isotonic regression calibrator for binary probabilities."""

    def __init__(self, y_min: float = 1e-6, y_max: float = 1.0 - 1e-6):
        self.y_min = y_min
        self.y_max = y_max
        self.model: Optional[IsotonicRegression] = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=self.y_min, y_max=self.y_max)
        self.model.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true, dtype=int))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator must be fit before transform")
        return np.clip(self.model.transform(np.asarray(y_prob, dtype=float)), self.y_min, self.y_max)

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        self.fit(y_prob, y_true)
        return self.transform(y_prob)
