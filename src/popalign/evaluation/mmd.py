"""Empirical maximum mean discrepancy with the linear kernel."""

import numpy as np


def mmd_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Return ||mean(X) - mean(Y)||₂ for k(x, y) = xᵀy.

    Uses all supplied rows. This is the distance between empirical kernel means;
    it measures mean differences and is zero for distributions with equal means.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if (X.ndim != 2 or Y.ndim != 2 or 0 in X.shape or 0 in Y.shape
            or X.shape[1] != Y.shape[1]):
        raise ValueError("MMD inputs must be nonempty 2D arrays with matching feature dimensions")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("MMD inputs must contain only finite values")
    return float(np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)))
