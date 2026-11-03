"""CPU alignment reference with camera-ready Eq. 3 top-fraction IS selection.

This implementation was added for the public release; it is not the script that
produced the paper's experiments. Defaults are small demonstration settings.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity


@dataclass
class AlignmentResult:
    """Row indices are zero-based; candidates are ranked, distinct input rows."""

    importance_probabilities: np.ndarray
    candidate_rows: np.ndarray
    cost: np.ndarray
    plan: np.ndarray
    transport_costs: np.ndarray
    transport_probabilities: np.ndarray
    selected_candidates: np.ndarray
    selected_rows: np.ndarray
    sinkhorn_iterations: int
    marginal_error: float


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or 0 in value.shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite two-dimensional matrix")
    return value


def sinkhorn(cost, *, epsilon=0.1, tolerance=1e-9, max_iterations=5000):
    """Solve entropic OT for uniform marginals using log-domain Sinkhorn updates.

    Return (plan, iterations, largest absolute marginal error). Failure to reach
    the requested tolerance raises an error rather than returning an unbalanced
    approximation. Dense storage is intentional; callers must bound its size.
    """
    cost = _matrix(cost, "cost")
    if (cost < 0).any():
        raise ValueError("cost must be nonnegative")
    _positive(epsilon, "epsilon")
    _positive(tolerance, "tolerance")
    _count(max_iterations, "max_iterations")
    n, m = cost.shape
    with np.errstate(over="ignore"):
        log_kernel = -cost / epsilon
    if not np.isfinite(log_kernel).all():
        raise ValueError("cost / epsilon overflowed; rescale the inputs or increase epsilon")
    log_a, log_b = -np.log(n), -np.log(m)
    log_v = np.zeros(m)
    error = np.inf
    for iteration in range(1, max_iterations + 1):
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            log_plan = log_kernel + log_u[:, None] + log_v[None, :]
            row_mass = np.exp(logsumexp(log_plan, axis=1))
            col_mass = np.exp(logsumexp(log_plan, axis=0))
            error = float(max(np.max(np.abs(row_mass - 1 / n)),
                              np.max(np.abs(col_mass - 1 / m))))
            if np.isfinite(error) and error <= tolerance:
                return np.exp(log_plan), iteration, error
    raise RuntimeError(
        f"Sinkhorn did not converge after {max_iterations} iterations "
        f"(marginal error {error:.3g}, tolerance {tolerance:.3g}); "
        "increase max_iterations/epsilon or check feature scaling"
    )


def align(persona, human, *, retain_fraction=0.7, n_output=128, bandwidth=0.5,
          epsilon=0.1, temperature=0.1, feature_weights=None, seed=0,
          max_cost_entries=2_000_000, tolerance=1e-9, max_iterations=5000,
          transport_cost_convention="main-text"):
    """Select top-weight candidates, then apply OT to scored response vectors.

    IS retains floor(retain_fraction * N) distinct rows, ranked by KDE density
    ratio (camera-ready Eq. 3). Ties preserve input order. Only the final OT draw
    uses replacement. Other defaults are demonstration settings. Feature weights
    affect only the OT cost, not Gaussian KDE. No standardization, missing-value
    imputation, human subsampling, or approximation is performed automatically.
    """
    if transport_cost_convention not in {"main-text", "appendix"}:
        raise ValueError("transport_cost_convention must be main-text or appendix")
    persona = _matrix(persona, "persona")
    human = _matrix(human, "human")
    if persona.shape[1] != human.shape[1]:
        raise ValueError("persona and human must have the same feature dimension")
    if (isinstance(retain_fraction, (bool, np.bool_)) or
            not np.isfinite(retain_fraction) or not 0 < retain_fraction <= 1):
        raise ValueError("retain_fraction must be finite and in (0, 1]")
    # Decimal fractions such as 0.7 must retain exactly floor(7*N/10), even
    # where floating multiplication would round just below an integer.
    n_candidates = int(len(persona) * Fraction(str(retain_fraction)))
    if n_candidates == 0:
        raise ValueError("retain_fraction retains zero candidates; provide more personas or increase the fraction")
    for value, name in [(n_output, "n_output"),
                        (max_cost_entries, "max_cost_entries"),
                        (max_iterations, "max_iterations")]:
        _count(value, name)
    for value, name in [(bandwidth, "bandwidth"), (epsilon, "epsilon"),
                        (temperature, "temperature"), (tolerance, "tolerance")]:
        _positive(value, name)
    if n_candidates * human.shape[0] > max_cost_entries:
        raise ValueError(
            f"Dense OT needs {n_candidates * human.shape[0]:,} cost entries, "
            f"exceeding max_cost_entries={max_cost_entries:,}. "
            "Choose smaller explicit inputs/settings; no automatic subsampling is used."
        )
    d = persona.shape[1]
    weights = np.ones(d) if feature_weights is None else np.asarray(feature_weights, dtype=float)
    if weights.shape != (d,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("feature_weights must contain one finite positive value per feature")
    rng = np.random.default_rng(seed)
    # Eq. 2: exact Gaussian KDE; sklearn's tolerances disable tree approximation.
    human_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, atol=0, rtol=0).fit(human)
    persona_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, atol=0, rtol=0).fit(persona)
    log_ratio = human_kde.score_samples(persona) - persona_kde.score_samples(persona)
    if not np.isfinite(log_ratio).all():
        raise ValueError("KDE density ratio is non-finite; check input magnitudes and bandwidth")
    probabilities = np.exp(log_ratio - logsumexp(log_ratio))
    probabilities /= probabilities.sum()
    # Eq. 3: rank log ratios directly to preserve order even if probabilities
    # underflow. Stable sorting breaks equal-weight ties by original row order.
    candidates = np.argsort(-log_ratio, kind="stable")[:n_candidates]
    weighted_persona = persona[candidates] * np.sqrt(weights)
    weighted_human = human * np.sqrt(weights)
    cost = cdist(weighted_persona, weighted_human, metric="sqeuclidean")
    plan, iterations, error = sinkhorn(cost, epsilon=epsilon, tolerance=tolerance,
                                       max_iterations=max_iterations)
    # Preserve the legacy OT conventions. The "appendix" option divides by
    # a_i = 1/n_candidates, as in the current camera-ready Eq. 6 / Eq. 28.
    transport_costs = np.sum(plan * cost, axis=1)
    if transport_cost_convention == "appendix":
        transport_costs *= n_candidates
    with np.errstate(over="ignore"):
        log_ot_weights = -transport_costs / temperature
    if not np.isfinite(log_ot_weights).all():
        raise ValueError("OT weights overflowed; check temperature and feature scaling")
    ot_probabilities = np.exp(log_ot_weights - logsumexp(log_ot_weights))
    ot_probabilities /= ot_probabilities.sum()
    selected = rng.choice(n_candidates, size=n_output, replace=True, p=ot_probabilities)
    return AlignmentResult(probabilities, candidates, cost, plan, transport_costs,
                           ot_probabilities, selected, candidates[selected], iterations, error)


__all__ = ["AlignmentResult", "align", "sinkhorn"]
