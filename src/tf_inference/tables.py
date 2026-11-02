"""Portable generated tables, with explicit generation and validation metadata."""

import gzip
import hashlib
import io
import json
from dataclasses import replace
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from .calibration import Calibration, CalibrationDiagnostics
from .curve import Curve, validate_alpha


def table_key(alpha: float) -> str:
    """Return a stable filename key without rounding distinct alpha values."""
    return repr(validate_alpha(alpha))


def available_alphas() -> tuple[float, ...]:
    """List significance levels with bundled precomputed calibrations."""
    data = files("tf_inference").joinpath("data")
    return tuple(
        sorted(
            float(p.name[6:-5])
            for p in data.iterdir()
            if p.name.startswith("alpha-") and p.name.endswith(".json")
        )
    )


def export_table(calibration: Calibration, directory, *, points: int = 16385):
    """Export NPZ, gzip CSV and JSON metadata, including interpolation error.

    The CSV contains F, the final critical value for absolute t, and the raw
    branch's regularized squared critical value. The NPZ stores the same branch
    arrays to evaluate close to the divergence.
    Its boundary row is the limiting regularized value, not a finite critical
    value. Accuracy is measured against the computed curve at intervening points;
    this does not certify error against the exact mathematical function.
    """
    if isinstance(points, bool) or not isinstance(points, int) or points < 128:
        raise ValueError("points must be an integer of at least 128.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    curve = calibration.curve
    q = curve.boundary
    endpoint = curve.property_c_endpoint
    distance = np.geomspace(1e-10, endpoint - q, points - 1)
    f = np.unique(np.r_[q, q + distance, calibration.plateau_f, endpoint])
    regularized = curve._spline(f)
    spline = PchipInterpolator(f, regularized, extrapolate=False)
    middle = f[:-1] + np.diff(f) / 2
    relative_error = float(
        np.max(np.abs(np.sqrt(spline(middle) / curve._spline(middle)) - 1))
    )
    if relative_error > 1e-6:
        raise ValueError("Table interpolation error exceeds 1e-6; increase points.")
    prefix = directory / f"alpha-{table_key(curve.alpha)}"
    npz = Path(str(prefix) + ".npz")
    np.savez_compressed(npz, f=f, regularized=regularized)
    csv_path = Path(str(prefix) + ".csv.gz")
    with gzip.GzipFile(filename=str(csv_path), mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            np.savetxt(
                stream,
                np.column_stack([f, calibration.critical_value(f), regularized]),
                delimiter=",",
                header="F,critical_value,branch_regularized_squared",
                comments="",
                fmt="%.17g",
            )
    metadata = {
        "schema_version": 1,
        "alpha": curve.alpha,
        "boundary": q,
        "epsilon": curve.epsilon,
        "seeds": curve.seeds,
        "points": len(f),
        "plateau_f": calibration.plateau_f,
        "minimum_plateau_f": curve.minimum_plateau_f,
        "property_c_endpoint": endpoint,
        "table_relative_error_on_midpoints": relative_error,
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "diagnostics": calibration.diagnostics.to_dict(),
        "reference": "https://doi.org/10.1257/aer.20211063",
        "algorithm": (
            "Appendix B.3; regularized inverse map; Property C; finite size search"
        ),
    }
    Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_table(alpha: float = 0.05, *, directory=None) -> Calibration:
    """Load a calibration, caching bundled tables across all public entry points.

    Custom directories are read and checksum-verified on every call. No
    interpolation between alpha levels is performed.
    """
    alpha = validate_alpha(alpha)
    if directory is None:
        return _load_bundled_table(alpha)
    return _read_table(alpha, Path(directory))


@lru_cache(maxsize=32)
def _load_bundled_table(alpha):
    return _read_table(alpha, files("tf_inference").joinpath("data"))


def _read_table(alpha, root):
    prefix = f"alpha-{table_key(alpha)}"
    metadata_path = root.joinpath(prefix + ".json")
    if not metadata_path.is_file():
        raise ValueError(
            f"No table for alpha={alpha}; use calibrate(alpha) or method='compute'."
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata["schema_version"] != 1 or metadata["alpha"] != float(alpha):
        raise ValueError("Unsupported or inconsistent table metadata.")
    npz = root.joinpath(prefix + ".npz")
    contents = npz.read_bytes()
    if hashlib.sha256(contents).hexdigest() != metadata["npz_sha256"]:
        raise ValueError("Table checksum does not match its metadata.")
    with np.load(io.BytesIO(contents), allow_pickle=False) as data:
        f, regularized = data["f"], data["regularized"]
    f.setflags(write=False)
    regularized.setflags(write=False)
    spline = PchipInterpolator(f, regularized, extrapolate=False)
    curve = Curve(
        metadata["alpha"],
        metadata["boundary"],
        f,
        regularized,
        metadata["epsilon"],
        metadata["seeds"],
        metadata["property_c_endpoint"],
        metadata["minimum_plateau_f"],
        spline,
    )
    diagnostics = CalibrationDiagnostics(**metadata["diagnostics"])
    diagnostics = replace(
        diagnostics,
        correlations=tuple(diagnostics.correlations),
        strengths=tuple(diagnostics.strengths),
    )
    return Calibration(curve, metadata["plateau_f"], diagnostics)
