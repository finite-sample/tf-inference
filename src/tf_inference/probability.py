"""Independent integration of the weak-IV Gaussian null experiment."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr, ndtri

from .curve import ConvergenceError, Curve


@dataclass(frozen=True)
class RejectionProbability:
    """Null rejection probability with quadrature and omitted-tail error estimates."""

    probability: float
    error: float


def _singular_roots(curve, plateau_f, strength):
    q = curve.boundary
    z = np.sqrt(q)
    plateau = curve.critical(plateau_f)

    def cv(f):
        if f * f <= q:
            return np.inf
        return curve.critical(min(f * f, plateau_f))

    upper = max(strength + plateau + z + 2, np.sqrt(plateau_f) + 1)
    left = brentq(lambda f: f * (f + strength) / strength - cv(f), z, upper)
    right = brentq(
        lambda f: f * (f - strength) / strength - cv(f),
        max(z, strength),
        upper,
    )
    roots = [-left, right]
    if strength > z:

        def inside(f):
            return f * (strength - f) / strength - cv(f)

        peak = minimize_scalar(
            lambda f: -inside(f), bounds=(z, strength), method="bounded"
        )
        if not peak.success:
            raise ConvergenceError("Could not locate the singular rejection region.")
        if inside(peak.x) > 0:
            roots.extend([brentq(inside, z, peak.x), brentq(inside, peak.x, strength)])
    return roots


def rejection_probability(
    curve: Curve,
    strength: float,
    rho: float,
    *,
    plateau_f: float | None = None,
    atol: float = 1e-9,
) -> RejectionProbability:
    """Integrate null rejection probability in the weak-IV limit experiment.

    ``strength`` is f0 (the mean first-stage t), not E[F]; E[F] = 1 + f0**2.
    ``rho`` is the correlation of the Gaussian AR and first-stage statistics.
    Symmetry allows their absolute values. This is asymptotic size, not a
    finite-sample coverage guarantee. The error excludes curve approximation.
    """
    strength, rho = float(strength), float(rho)
    if not np.isfinite(strength) or not np.isfinite(rho) or abs(rho) > 1:
        raise ValueError("strength must be finite and rho must lie in [-1, 1].")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive.")
    strength, rho = abs(strength), abs(rho)
    stop = curve.minimum_plateau_f if plateau_f is None else float(plateau_f)
    if not curve.boundary < stop <= curve.property_c_endpoint:
        raise ValueError("plateau_f must lie within the Property (C) domain.")
    plateau = curve.critical(stop)
    z = np.sqrt(curve.boundary)
    if rho == 1:
        if strength == 0:
            return RejectionProbability(curve.alpha, 0.0)
        roots = _singular_roots(curve, stop, strength)
        value = ndtr(roots[0] - strength) + ndtr(strength - roots[1])
        if len(roots) == 4:
            value += ndtr(roots[3] - strength) - ndtr(roots[2] - strength)
        root_error = sum(
            2e-12 + 4 * np.finfo(float).eps * abs(root) for root in roots
        ) / np.sqrt(2 * np.pi)
        return RejectionProbability(float(value), float(root_error))

    sigma = np.sqrt((1 - rho) * (1 + rho))

    def integrand(f):
        f_stat = f * f
        if f_stat <= curve.boundary:
            return 0.0
        critical_sq = (
            plateau**2 if f_stat >= stop else float(curve.critical(f_stat)) ** 2
        )
        a = f_stat - critical_sq
        half_b = critical_sq * rho * f
        discriminant = critical_sq * f_stat * (f_stat - critical_sq * sigma**2)
        if discriminant <= 0:
            return 0.0
        mean = rho * (f - strength)
        if abs(a) < 1e-12 * max(f_stat, critical_sq):
            cutoff = critical_sq * f_stat / (2 * half_b)
            probability = ndtr((mean - cutoff) / sigma)
            if half_b < 0:
                probability = ndtr((cutoff - mean) / sigma)
        else:
            root = (-half_b - np.copysign(np.sqrt(discriminant), half_b)) / a
            other = -critical_sq * f_stat / (a * root)
            low, high = sorted((root, other))
            if a > 0:
                probability = ndtr((low - mean) / sigma) + ndtr((mean - high) / sigma)
            elif low > mean:
                probability = ndtr((mean - low) / sigma) - ndtr((mean - high) / sigma)
            else:
                probability = ndtr((high - mean) / sigma) - ndtr((low - mean) / sigma)
        density = np.exp(-0.5 * (f - strength) ** 2) / np.sqrt(2 * np.pi)
        return float(probability * density)

    tail = max(9.0, -float(ndtri(atol / 100)))
    low, high = strength - tail, strength + tail
    splits = [low, high, -z, z, -np.sqrt(stop), np.sqrt(stop), strength]
    # Split where the quadratic discriminant vanishes, and at the limiting
    # rho=1 roots; these locate increasingly narrow transitions as rho -> 1.
    discriminant_root = brentq(
        lambda f: f * f - float(curve.critical(min(f * f, stop))) ** 2 * sigma**2,
        z,
        max(np.sqrt(stop), plateau + z + 1),
    )
    splits.extend([-discriminant_root, discriminant_root])
    roots = []
    if strength > 0:
        roots = _singular_roots(curve, stop, strength)
        splits.extend(roots)
    if rho > 0.95:
        for root in [*roots, -z, z, -discriminant_root, discriminant_root]:
            for width in (sigma, sigma**2):
                for scale in (0.01, 0.1, 1.0, 10.0):
                    splits.extend([root - scale * width, root + scale * width])
    splits = sorted(set(value for value in splits if low <= value <= high))
    value, error = 0.0, float(2 * ndtr(-tail))
    for lower, upper in zip(splits[:-1], splits[1:], strict=True):
        result = quad(
            integrand,
            lower,
            upper,
            epsabs=atol / len(splits),
            epsrel=1e-7,
            limit=250,
            full_output=1,
        )
        if len(result) != 3:
            raise ConvergenceError(
                f"Rejection probability integration failed: {result[3]}"
            )
        value += result[0]
        error += result[1]
    return RejectionProbability(float(value), float(error))
