"""The Appendix B.3 inverse map and the Property (C) endpoint."""

from dataclasses import dataclass
from math import isfinite
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeResult, brentq, minimize_scalar
from scipy.special import ndtr, ndtri


class ConvergenceError(RuntimeError):
    """A numerical construction or calibration did not converge."""


def validate_alpha(alpha: float) -> float:
    """Validate the numerical implementation's significance-level domain."""
    if np.iscomplexobj(alpha):
        raise ValueError("alpha must be real.")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0001 <= alpha <= 0.25:
        raise ValueError("alpha must be finite and between 0.0001 and 0.25.")
    return alpha


def inverse_map(f_stat: ArrayLike, critical: ArrayLike, alpha: float):
    """Extend the decreasing branch using Appendix B.3, step 2."""
    f_stat, critical = np.broadcast_arrays(f_stat, critical)
    strength = f_stat / (critical - np.sqrt(f_stat))
    # Survival probabilities avoid subtracting a small tail from one.
    normal_quantile = -ndtri(alpha - ndtr(-np.sqrt(f_stat) - strength))
    root_f = strength + normal_quantile
    return root_f**2, root_f * normal_quantile / strength


@dataclass(frozen=True)
class Curve:
    """A numerically constructed decreasing branch, before plateau calibration.

    ``critical`` returns the threshold for absolute t, not for t squared.
    The interpolated quantity is (F - boundary) * critical(F)**2.
    """

    alpha: float
    boundary: float
    f: np.ndarray
    regularized: np.ndarray
    epsilon: float
    seeds: int
    property_c_endpoint: float
    minimum_plateau_f: float
    _spline: PchipInterpolator

    def regularized_squared(self, f_stat: ArrayLike):
        """Evaluate the finite regularized squared critical value.

        Args:
            f_stat: Real F values between the boundary and the branch endpoint.

        Returns:
            Values of (F - boundary) times the squared critical value.

        Raises:
            ValueError: If F is non-real, nonfinite, or outside the branch.
        """
        if np.iscomplexobj(f_stat):
            raise ValueError("F must be real.")
        values = np.asarray(f_stat, dtype=float)
        if (
            np.any(~np.isfinite(values))
            or np.any(values < self.boundary)
            or np.any(values > self.f[-1])
        ):
            raise ValueError("F lies outside the constructed decreasing branch.")
        return self._spline(values)

    def critical(self, f_stat: ArrayLike):
        """Evaluate the decreasing branch within its constructed domain."""
        if np.iscomplexobj(f_stat):
            raise ValueError("F must be real.")
        values = np.asarray(f_stat, dtype=float)
        if values.ndim == 0:
            value = float(values)
            if not isfinite(value) or value < 0:
                raise ValueError("F must be finite and nonnegative.")
            if value > self.f[-1]:
                raise ValueError("F lies outside the constructed decreasing branch.")
            if value <= self.boundary:
                return float("inf")
            return float(np.sqrt(self._spline(value) / (value - self.boundary)))
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("F must be finite and nonnegative.")
        if np.any(values > self.f[-1]):
            raise ValueError("F lies outside the constructed decreasing branch.")
        result = np.full(values.shape, np.inf)
        finite = values > self.boundary
        result[finite] = np.sqrt(
            self._spline(values[finite]) / (values[finite] - self.boundary)
        )
        return float(result) if result.ndim == 0 else result


def construct_curve(
    alpha: float = 0.05, *, epsilon: float = 1e-4, seeds: int = 256
) -> Curve:
    """Construct Appendix B.3's decreasing branch and its admissible endpoint.

    Args:
        alpha: Two-sided significance level, between 0.0001 and 0.25.
        epsilon: Initialization distance from the chi-square boundary.
        seeds: Number of starting points across one inverse-map interval.

    Returns:
        Candidate decreasing curve and its admissible endpoint. Use ``calibrate``
        to check a plateau before using the curve for inference.

    Raises:
        ValueError: If the significance level or construction settings are invalid.
        ConvergenceError: If the recurrence or tangency calculation fails.
    """
    alpha = validate_alpha(alpha)
    if np.iscomplexobj(epsilon):
        raise ValueError("epsilon must be real.")
    if not np.isfinite(epsilon) or not 1e-6 <= epsilon <= 1e-3:
        raise ValueError("epsilon must be between 1e-6 and 1e-3.")
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 32:
        raise ValueError("seeds must be an integer of at least 32.")
    q = float(ndtri(alpha / 2) ** 2)
    b = 3 * q - q**2 / 2 + q**3 / 6
    start = q + epsilon
    end = float(inverse_map(start, np.sqrt(q**3 / (start - q) - b), alpha)[0])
    f = np.linspace(start, end, seeds, endpoint=False)
    critical = np.sqrt(q**3 / (f - q) - b)
    f_parts, c_parts = [], []
    for _ in range(10000):
        f_parts.append(f)
        c_parts.append(critical)
        valid = (critical > np.sqrt(f)) & (f < 1e5)
        if not np.any(valid):
            break
        next_f, next_c = inverse_map(f[valid], critical[valid], alpha)
        if np.any(~np.isfinite(next_f)) or np.any(next_f <= f[valid]):
            raise ConvergenceError("The inverse map failed to extend its domain.")
        f, critical = next_f, next_c
    else:
        raise ConvergenceError("The inverse map exceeded 10000 iterations.")
    f, critical = np.concatenate(f_parts), np.concatenate(c_parts)
    order = np.argsort(f)
    f, critical = f[order], critical[order]
    unique = np.r_[True, np.diff(f) > 0]
    f, critical = f[unique], critical[unique]
    if np.any(np.diff(critical) > 1e-8 * critical[:-1]):
        raise ConvergenceError(
            "The constructed critical-value branch is not decreasing."
        )
    regularized = np.r_[q**3, (f - q) * critical**2]
    f = np.r_[q, f]
    spline = PchipInterpolator(f, regularized, extrapolate=False)

    def cv(value):
        return np.sqrt(spline(value) / (value - q))

    # An interior rejection interval appears when strength exceeds min_F
    # F / (sqrt(F) - critical(F)). This is the tangency in Property (C).
    lower = f[1:][f[1:] > critical**2][0]
    upper = min(1e4, f[-1])
    contact = cast(
        "OptimizeResult",
        minimize_scalar(
            lambda value: value / (np.sqrt(value) - cv(value)),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-9},
        ),
    )
    if not contact.success:
        raise ConvergenceError("Could not locate the Property (C) tangency.")
    endpoint = cast(
        "float",
        brentq(
            lambda value: value / (np.sqrt(value) + cv(value)) - contact.fun,
            contact.x,
            upper,
        ),
    )
    plateau_f = endpoint
    if cv(endpoint) < np.sqrt(q):
        plateau_f = cast(
            "float", brentq(lambda value: cv(value) - np.sqrt(q), q + epsilon, endpoint)
        )
    f.setflags(write=False)
    regularized.setflags(write=False)
    return Curve(alpha, q, f, regularized, epsilon, seeds, endpoint, plateau_f, spline)
