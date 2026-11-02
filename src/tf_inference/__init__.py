"""Numerically constructed tF inference for single-instrument IV models."""

from functools import lru_cache

from .calibration import Calibration, CalibrationDiagnostics, calibrate
from .curve import ConvergenceError, Curve, construct_curve
from .probability import RejectionProbability, rejection_probability
from .tables import available_alphas, export_table, load_table

__all__ = [
    "Calibration",
    "CalibrationDiagnostics",
    "ConvergenceError",
    "Curve",
    "RejectionProbability",
    "available_alphas",
    "calibrate",
    "confidence_interval",
    "construct_curve",
    "critical_value",
    "export_table",
    "load_table",
    "rejection_probability",
]


@lru_cache(maxsize=32)
def _computed_calibration(alpha):
    return calibrate(alpha)


def _calibration(alpha, method):
    if method == "table":
        return load_table(alpha)
    if method == "compute":
        return _computed_calibration(alpha)
    raise ValueError("method must be 'table' or 'compute'.")


def critical_value(f_stat, alpha=0.05, *, method="table"):
    """Return tF critical values for absolute t using a table or numerical construction.

    The default requires an exact matching bundled alpha. Explicitly select
    ``method="compute"`` for cached numerical calibration at another level.
    No interpolation across significance levels is performed.
    For custom numerical tolerances use ``calibrate`` and inspect its diagnostics.
    """
    return _calibration(float(alpha), method).critical_value(f_stat)


def confidence_interval(coefficient, se, f_stat, alpha=0.05, *, method="table"):
    """Return tF confidence intervals with lower/upper bounds on the last axis."""
    return _calibration(float(alpha), method).confidence_interval(
        coefficient, se, f_stat
    )
