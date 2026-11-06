"""Independent integration of the weak-IV Gaussian null experiment."""

import itertools
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.integrate import quad
from scipy.optimize import OptimizeResult, brentq, minimize_scalar
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
    left = cast(
        "float", brentq(lambda f: f * (f + strength) / strength - cv(f), z, upper)
    )
    right = cast(
        "float",
        brentq(
            lambda f: f * (f - strength) / strength - cv(f),
            max(z, strength),
            upper,
        ),
    )
    roots = [-left, right]
    if strength > z:

        def inside(f):
            return f * (strength - f) / strength - cv(f)

        peak = cast(
            "OptimizeResult",
            minimize_scalar(
                lambda f: -inside(f), bounds=(z, strength), method="bounded"
            ),
        )
        if not peak.success:
            raise ConvergenceError("Could not locate the singular rejection region.")
        if inside(peak.x) > 0:
            roots.extend(
                [
                    cast("float", brentq(inside, z, peak.x)),
                    cast("float", brentq(inside, peak.x, strength)),
                ]
            )
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
    if any(np.iscomplexobj(value) for value in (strength, rho, plateau_f, atol)):
        raise ValueError("strength, rho, plateau_f and atol must be real.")
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
    tail = max(9.0, -float(ndtri(atol / 100)))
    if strength > max(2 * tail, np.sqrt(stop) + tail):
        # Restrict both standard-normal innovations to [-tail, tail]. On this
        # event the t denominator lies between 1-d and 1+d, d=tail/(strength-tail).
        # The normal density bounds the resulting change in tail probability.
        bound = (
            2 * plateau * (tail / (strength - tail)) / np.sqrt(2 * np.pi)
            + 4 * ndtr(-tail)
            + 16 * np.finfo(float).eps
        )
        if bound <= atol / 10:
            return RejectionProbability(float(2 * ndtr(-plateau)), float(bound))
    if rho == 1:
        if strength == 0:
            return RejectionProbability(curve.alpha, 0.0)
        if strength - tail > np.sqrt(stop) and strength > 4 * plateau:
            upper = 2 * plateau / (np.sqrt(1 + 4 * plateau / strength) + 1)
            lower = -2 * plateau / (np.sqrt(1 - 4 * plateau / strength) + 1)
            value = (
                ndtr(-upper)
                + ndtr(lower)
                + ndtr(-strength - upper)
                - ndtr(-strength - lower)
            )
            error = 2 * ndtr(-tail) + 16 * np.finfo(float).eps
            return RejectionProbability(float(value), float(error))
        roots = _singular_roots(curve, stop, strength)
        value = ndtr(roots[0] - strength) + ndtr(strength - roots[1])
        if len(roots) == 4:
            value += ndtr(roots[3] - strength) - ndtr(roots[2] - strength)
        root_error = sum(
            2e-12 + 4 * np.finfo(float).eps * abs(root) for root in roots
        ) / np.sqrt(2 * np.pi)
        return RejectionProbability(float(value), float(root_error))

    sigma = np.sqrt((1 - rho) * (1 + rho))

    def integrand(innovation):
        f = strength + innovation
        if abs(f) <= z:
            return 0.0
        critical = plateau if abs(f) >= np.sqrt(stop) else float(curve.critical(f * f))
        ratio = critical / f
        a = 1 - ratio**2
        half_b = critical * ratio * rho
        discriminant = critical**2 * (1 - ratio**2 * sigma**2)
        if discriminant <= 0:
            return 0.0
        mean = rho * innovation
        if abs(a) < 1e-12 * max(1, ratio**2):
            cutoff = critical**2 / (2 * half_b)
            probability = ndtr((mean - cutoff) / sigma)
            if half_b < 0:
                probability = ndtr((cutoff - mean) / sigma)
        else:
            root = (-half_b - np.copysign(np.sqrt(discriminant), half_b)) / a
            other = -(critical**2) / (a * root)
            low, high = sorted((root, other))
            if a > 0:
                probability = ndtr((low - mean) / sigma) + ndtr((mean - high) / sigma)
            elif low > mean:
                probability = ndtr((mean - low) / sigma) - ndtr((mean - high) / sigma)
            else:
                probability = ndtr((high - mean) / sigma) - ndtr((low - mean) / sigma)
        density = np.exp(-0.5 * innovation**2) / np.sqrt(2 * np.pi)
        return float(probability * density)

    low, high = -tail, tail
    splits = [
        low,
        high,
        -z - strength,
        z - strength,
        -np.sqrt(stop) - strength,
        np.sqrt(stop) - strength,
        0.0,
    ]
    # Split where the quadratic discriminant vanishes, and at the limiting
    # rho=1 roots; these locate increasingly narrow transitions as rho -> 1.
    discriminant_root = cast(
        "float",
        brentq(
            lambda f: f * f - float(curve.critical(min(f * f, stop))) ** 2 * sigma**2,
            z,
            max(np.sqrt(stop), plateau + z + 1),
        ),
    )
    splits.extend([-discriminant_root - strength, discriminant_root - strength])
    roots = []
    if strength - tail > np.sqrt(stop) and strength > 4 * plateau:
        roots = [
            2 * plateau / (np.sqrt(1 + 4 * plateau / strength) + 1),
            -2 * plateau / (np.sqrt(1 - 4 * plateau / strength) + 1),
        ]
        splits.extend(roots)
    elif strength > 0:
        roots = [value - strength for value in _singular_roots(curve, stop, strength)]
        splits.extend(roots)
    if rho > 0.95:
        for root in [
            *roots,
            -z - strength,
            z - strength,
            -discriminant_root - strength,
            discriminant_root - strength,
        ]:
            for width in (sigma, sigma**2):
                for scale in (0.01, 0.1, 1.0, 10.0):
                    splits.extend([root - scale * width, root + scale * width])
    splits = sorted({value for value in splits if low <= value <= high})
    value, error = 0.0, float(2 * ndtr(-tail))
    for lower, upper in itertools.pairwise(splits):
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
