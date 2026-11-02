"""Numerical plateau selection, audit records, and inference."""

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq, minimize, minimize_scalar

from .curve import ConvergenceError, Curve, construct_curve
from .probability import rejection_probability


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Finite numerical size search; this is not a uniform mathematical bound."""

    alpha: float
    plateau_f: float
    plateau_critical: float
    property_c_endpoint: float
    maximum_rejection: float
    worst_strength: float
    worst_rho: float
    integration_error: float
    size_tolerance: float
    strength_limit: float
    evaluations: int
    adjustments: int
    correlations: tuple[float, ...]
    strengths: tuple[float, ...]

    def to_dict(self):
        """Return serializable diagnostics."""
        return asdict(self)


@dataclass(frozen=True)
class Calibration:
    """A tF curve and a plateau checked by a finite numerical size search."""

    curve: Curve
    plateau_f: float
    diagnostics: CalibrationDiagnostics

    @property
    def alpha(self):
        """Two-sided significance level."""
        return self.curve.alpha

    @property
    def boundary(self):
        """F at or below which the confidence interval is unbounded."""
        return self.curve.boundary

    @property
    def plateau(self):
        """Critical value for absolute t on the plateau."""
        return float(self.curve.critical(self.plateau_f))

    def critical_value(self, f_stat: ArrayLike):
        """Evaluate critical values, with infinity at and below the boundary."""
        values = np.asarray(f_stat, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("F must be finite and nonnegative.")
        return self.curve.critical(np.minimum(values, self.plateau_f))

    def confidence_interval(
        self, coefficient: ArrayLike, se: ArrayLike, f_stat: ArrayLike
    ):
        """Construct intervals; broadcast inputs and put lower/upper bounds last."""
        coefficient, se, f_stat = np.broadcast_arrays(coefficient, se, f_stat)
        if (
            np.any(~np.isfinite(coefficient))
            or np.any(~np.isfinite(se))
            or np.any(se <= 0)
        ):
            raise ValueError("coefficient must be finite and se finite and positive.")
        width = self.critical_value(f_stat) * se
        return np.stack([coefficient - width, coefficient + width], axis=-1)


def calibrate(
    alpha: float = 0.05,
    *,
    epsilon: float = 1e-4,
    seeds: int = 256,
    size_tolerance: float = 2e-7,
    strength_limit: float = 80.0,
    correlations: ArrayLike | None = None,
    strengths: ArrayLike | None = None,
    optimize: bool = True,
) -> Calibration:
    """Construct and numerically calibrate a tF curve at a requested alpha.

    The initial plateau satisfies Property (C) and is at least the normal
    critical value. A grid and local searches check rejection probabilities;
    violations move the plateau earlier on the decreasing branch. Diagnostics
    report the finite search domain and numerical tolerances. Neither a finite
    grid nor local optimization certifies the supremum over all nuisance values.
    """
    if not np.isfinite(size_tolerance) or not 1e-9 <= size_tolerance <= 1e-3:
        raise ValueError("size_tolerance must be between 1e-9 and 1e-3.")
    if not np.isfinite(strength_limit) or strength_limit < 20:
        raise ValueError("strength_limit must be finite and at least 20.")
    curve = construct_curve(alpha, epsilon=epsilon, seeds=seeds)
    stop = curve.minimum_plateau_f
    if correlations is None:
        correlations = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999, 1]
    rho_grid = np.unique(np.r_[correlations, 0.0, 1.0]).astype(float)
    if np.any(~np.isfinite(rho_grid)) or np.any((rho_grid < 0) | (rho_grid > 1)):
        raise ValueError("correlations must be finite and lie in [0, 1].")
    if strengths is None:
        strengths = np.r_[0, 0.05, 0.1, 0.25, 0.5, np.geomspace(1, strength_limit, 32)]
    m_grid = np.unique(np.r_[strengths, 0.0, strength_limit]).astype(float)
    if np.any(~np.isfinite(m_grid)) or np.any((m_grid < 0) | (m_grid > strength_limit)):
        raise ValueError("strengths must be finite and lie in [0, strength_limit].")
    integration_tol = min(1e-9, size_tolerance / 50)
    evaluations = 0
    max_error = 0.0

    def evaluate(m, rho, candidate):
        nonlocal evaluations, max_error
        result = rejection_probability(
            curve, m, rho, plateau_f=candidate, atol=integration_tol
        )
        evaluations += 1
        max_error = max(max_error, result.error)
        if result.error > size_tolerance / 5:
            raise ConvergenceError(
                "Quadrature error is too large for size calibration."
            )
        return result.probability

    def search(candidate):
        records = [
            (evaluate(m, rho, candidate), float(m), float(rho))
            for m in m_grid
            for rho in rho_grid
        ]
        if optimize:
            # Refine each resolved peak along the singular boundary, including
            # the post-plateau hump which a coarse strength grid can miss.
            singular = sorted((p, m) for p, m, r in records if r == 1)
            for _, m in singular[-3:]:
                lo, hi = max(0.001, m / 1.5), min(strength_limit, max(m + 1, m * 1.5))
                result = minimize_scalar(
                    lambda x: -evaluate(x, 1.0, candidate),
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-5},
                )
                if result.success:
                    records.append((-float(result.fun), float(result.x), 1.0))
            # Also search the interior; rho=1 is not assumed to be worst-case.
            interior = sorted(record for record in records if record[2] < 0.999)
            for _, m, rho in interior[-2:]:
                result = minimize(
                    lambda x: -evaluate(float(x[0]), float(x[1]), candidate),
                    [m, rho],
                    method="Nelder-Mead",
                    bounds=[(0.0, strength_limit), (0.0, 0.99999)],
                    options={
                        "maxiter": 150,
                        "xatol": 1e-4,
                        "fatol": size_tolerance / 10,
                    },
                )
                if not result.success:
                    raise ConvergenceError("The interior size search did not converge.")
                records.append(
                    (-float(result.fun), float(result.x[0]), float(result.x[1]))
                )
        return max(records)

    for adjustments in range(12):
        worst, m, rho = search(stop)
        if worst <= curve.alpha + size_tolerance:
            diagnostics = CalibrationDiagnostics(
                curve.alpha,
                stop,
                float(curve.critical(stop)),
                curve.property_c_endpoint,
                worst,
                m,
                rho,
                max_error,
                size_tolerance,
                strength_limit,
                evaluations,
                adjustments,
                tuple(rho_grid),
                tuple(m_grid),
            )
            return Calibration(curve, stop, diagnostics)
        target = curve.alpha - integration_tol * 10
        lower = curve.boundary + epsilon
        if evaluate(m, rho, lower) > target:
            raise ConvergenceError("No admissible plateau resolves the size violation.")
        stop = brentq(
            lambda value, m=m, rho=rho, target=target: evaluate(m, rho, value) - target,
            lower,
            stop,
        )
    raise ConvergenceError("Plateau calibration exceeded 12 adjustments.")
