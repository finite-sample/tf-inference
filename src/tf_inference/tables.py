"""Schema-validated Parquet calibration tables and optional CSV export."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from dataclasses import asdict, replace
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtri

from tf_inference.calibration import Calibration, CalibrationDiagnostics
from tf_inference.curve import Curve, validate_alpha

TABLE_SCHEMA = pa.schema(
    [
        pa.field("alpha", pa.float64(), nullable=False),
        pa.field("f_stat", pa.float64(), nullable=False),
        pa.field("regularized_squared", pa.float64(), nullable=False),
    ]
)
_DATA_FILE = "calibrations.parquet"
_MANIFEST_FILE = "calibrations.json"


def available_alphas() -> tuple[float, ...]:
    """List significance levels with bundled precomputed calibrations.

    Returns:
        Sorted significance levels from the validated table metadata.
    """
    _, metadata = _bundled_dataset()
    return tuple(sorted(float(key) for key in metadata))


def _read_dataset(root: Any) -> tuple[pa.Table, dict[str, Any]]:
    manifest = json.loads(root.joinpath(_MANIFEST_FILE).read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise ValueError("Unsupported table manifest schema.")
    contents = root.joinpath(_DATA_FILE).read_bytes()
    if hashlib.sha256(contents).hexdigest() != manifest.get("parquet_sha256"):
        raise ValueError("Table checksum does not match its manifest.")
    table = pq.read_table(pa.BufferReader(contents))
    if not table.schema.equals(TABLE_SCHEMA, check_metadata=False):
        raise ValueError("Table columns do not match the declared Arrow schema.")
    if any(column.null_count for column in table.columns):
        raise ValueError("Table columns must not contain nulls.")
    if any(not np.isfinite(column.to_numpy()).all() for column in table.columns):
        raise ValueError("Table columns must contain finite real values.")
    raw_metadata = (table.schema.metadata or {}).get(b"tf_inference")
    if raw_metadata is None:
        raise ValueError("Table calibration metadata is missing.")
    metadata = json.loads(raw_metadata)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Table calibration metadata must be a nonempty mapping.")
    alphas = tuple(sorted(float(key) for key in metadata))
    if len(set(alphas)) != len(alphas) or any(
        repr(alpha) not in metadata for alpha in alphas
    ):
        raise ValueError(
            "Table metadata contains duplicate or noncanonical alpha keys."
        )
    if set(table.column("alpha").to_pylist()) != set(alphas):
        raise ValueError("Table significance levels do not match the metadata.")
    for alpha in alphas:
        _from_dataset(alpha, table, metadata)
    return table, metadata


@lru_cache(maxsize=1)
def _bundled_dataset() -> tuple[pa.Table, dict[str, Any]]:
    return _read_dataset(files("tf_inference").joinpath("data"))


def _validate_decreasing_spline(spline: PchipInterpolator, boundary: float) -> None:
    """Check the sign of the squared critical value's derivative on every cell."""
    a, b, c, d = spline.c
    width = np.diff(spline.x)
    offset = spline.x[:-1] - boundary
    stationary = np.zeros_like(a)
    np.divide(-b, 3 * a, out=stationary, where=a != 0)
    stationary = np.clip(stationary, 0, width)
    # For a local cubic Z(t), the derivative has the sign of
    # Q=(t+offset)*Z'-Z. Q'=2*(t+offset)*(3*a*t+b), so its only possible
    # interior extremum is -b/(3*a); the other root is outside the cell.
    for t in (np.zeros_like(width), width, stationary):
        value = ((a * t + b) * t + c) * t + d
        scaled_derivative = (t + offset) * ((3 * a * t + 2 * b) * t + c)
        tolerance = 1e-12 * (np.abs(value) + np.abs(scaled_derivative))
        if np.any(scaled_derivative - value > tolerance):
            raise ValueError("Calibration interpolation must be decreasing.")


def _from_dataset(
    alpha: float, table: pa.Table, metadata: dict[str, Any]
) -> Calibration:
    if repr(alpha) not in metadata:
        raise ValueError(
            f"No table for alpha={alpha}; use calibrate(alpha) or method='compute'."
        )
    record = metadata[repr(alpha)]
    if not isinstance(record, dict):
        raise ValueError("Calibration metadata must be an object.")
    required = {
        "alpha",
        "boundary",
        "epsilon",
        "seeds",
        "points",
        "plateau_f",
        "minimum_plateau_f",
        "property_c_endpoint",
        "diagnostics",
        "table_relative_error_on_midpoints",
    }
    if not required <= record.keys():
        raise ValueError("Calibration metadata is missing required fields.")
    validate_alpha(alpha)
    scalar_fields = (
        "alpha",
        "boundary",
        "epsilon",
        "plateau_f",
        "minimum_plateau_f",
        "property_c_endpoint",
        "table_relative_error_on_midpoints",
    )
    if any(
        isinstance(record[key], bool)
        or not isinstance(record[key], (int, float))
        or not np.isfinite(record[key])
        for key in scalar_fields
    ):
        raise ValueError("Calibration metadata requires finite numeric scalars.")
    if any(
        isinstance(record[key], bool) or not isinstance(record[key], int)
        for key in ("points", "seeds")
    ):
        raise ValueError("Calibration points and seeds must be integers.")
    rows = table.filter(pa.array(table.column("alpha").to_numpy() == alpha))
    f = rows.column("f_stat").to_numpy().copy()
    regularized = rows.column("regularized_squared").to_numpy().copy()
    boundary = record["boundary"]
    theoretical_boundary = float(ndtri(alpha / 2) ** 2)
    if (
        record["alpha"] != alpha
        or not np.isclose(
            boundary, theoretical_boundary, rtol=16 * np.finfo(float).eps, atol=0
        )
        or len(f) != record["points"]
        or len(f) < 2
        or f[0] != boundary
        or not np.all(np.diff(f) > 0)
        or not np.all(regularized > 0)
        or not np.isclose(regularized[0], boundary**3, rtol=1e-12)
        or not boundary
        < record["plateau_f"]
        <= record["minimum_plateau_f"]
        <= record["property_c_endpoint"]
        == f[-1]
        or not 1e-6 <= record["epsilon"] <= 1e-3
        or record["seeds"] < 32
        or not 0 <= record["table_relative_error_on_midpoints"] <= 1e-6
    ):
        raise ValueError(
            "Calibration table or metadata violates its numerical invariants."
        )
    critical = np.sqrt(regularized[1:] / (f[1:] - boundary))
    if np.any(np.diff(critical) > 1e-12 * critical[:-1]):
        raise ValueError("Calibration knots must describe a decreasing branch.")
    f.setflags(write=False)
    regularized.setflags(write=False)
    spline = PchipInterpolator(f, regularized, extrapolate=False)
    _validate_decreasing_spline(spline, boundary)
    curve = Curve(
        alpha,
        boundary,
        f,
        regularized,
        record["epsilon"],
        record["seeds"],
        record["property_c_endpoint"],
        record["minimum_plateau_f"],
        spline,
    )
    try:
        diagnostics = CalibrationDiagnostics(**record["diagnostics"])
        diagnostics = replace(
            diagnostics,
            correlations=tuple(diagnostics.correlations),
            strengths=tuple(diagnostics.strengths),
        )
    except (TypeError, KeyError) as error:
        raise ValueError("Invalid calibration diagnostics.") from error
    numeric = asdict(diagnostics)
    for key in ("correlations", "strengths"):
        values = numeric.pop(key)
        if not values or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            for value in values
        ):
            raise ValueError("Calibration diagnostics require finite numeric grids.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        for value in numeric.values()
    ):
        raise ValueError("Calibration diagnostics require finite numeric scalars.")
    if (
        not 0 <= diagnostics.maximum_rejection <= alpha + diagnostics.size_tolerance
        or not 0 <= diagnostics.worst_rho <= 1
        or not 0 <= diagnostics.worst_strength <= diagnostics.strength_limit
        or not 0 <= diagnostics.integration_error <= diagnostics.size_tolerance / 5
        or not 1e-9 <= diagnostics.size_tolerance <= 1e-3
        or diagnostics.strength_limit < 20
        or not isinstance(diagnostics.evaluations, int)
        or diagnostics.evaluations < 1
        or not isinstance(diagnostics.adjustments, int)
        or not 0 <= diagnostics.adjustments < 12
        or any(not 0 <= value <= 1 for value in diagnostics.correlations)
        or any(
            not 0 <= value <= diagnostics.strength_limit
            for value in diagnostics.strengths
        )
    ):
        raise ValueError("Calibration diagnostics violate their numerical domains.")
    if (
        diagnostics.plateau_critical < np.sqrt(theoretical_boundary) * (1 - 1e-12)
        or diagnostics.alpha != alpha
        or diagnostics.plateau_f != record["plateau_f"]
        or diagnostics.property_c_endpoint != record["property_c_endpoint"]
        or not np.isclose(
            diagnostics.plateau_critical,
            curve.critical(record["plateau_f"]),
            rtol=1e-12,
        )
    ):
        raise ValueError("Calibration diagnostics disagree with the table metadata.")
    return Calibration(curve, record["plateau_f"], diagnostics)


def _write_dataset(table: pa.Table, metadata: dict[str, Any], directory: Path) -> None:
    payload = json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    table = table.replace_schema_metadata({b"tf_inference": payload})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd", use_dictionary=True)
    contents = sink.getvalue().to_pybytes()
    directory.mkdir(parents=True, exist_ok=True)
    directory.joinpath(_DATA_FILE).write_bytes(contents)
    manifest = {
        "schema_version": 2,
        "parquet_sha256": hashlib.sha256(contents).hexdigest(),
    }
    directory.joinpath(_MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")


def export_table(
    calibration: Calibration,
    directory: str | Path,
    *,
    points: int = 16385,
    include_csv: bool = False,
) -> dict[str, Any]:
    """Export a calibration into a schema-validated Parquet dataset.

    Args:
        calibration: Numerically calibrated tF curve.
        directory: Output directory. Other significance levels are preserved.
        points: Number of grid points, at least 128.
        include_csv: Also write a gzip CSV for use in other software.

    Returns:
        Calibration metadata including measured midpoint interpolation error.

    Raises:
        ValueError: If points are invalid or interpolation error exceeds 1e-6.
    """
    if isinstance(points, bool) or not isinstance(points, int) or points < 128:
        raise ValueError("points must be an integer of at least 128.")
    directory = Path(directory)
    curve = calibration.curve
    boundary = curve.boundary
    distance = np.geomspace(1e-10, curve.property_c_endpoint - boundary, points - 1)
    f = np.unique(
        np.r_[
            boundary,
            boundary + distance,
            calibration.plateau_f,
            curve.property_c_endpoint,
        ]
    )
    regularized = curve.regularized_squared(f)
    spline = PchipInterpolator(f, regularized, extrapolate=False)
    middle = f[:-1] + np.diff(f) / 2
    relative_error = float(
        np.max(np.abs(np.sqrt(spline(middle) / curve.regularized_squared(middle)) - 1))
    )
    if relative_error > 1e-6:
        raise ValueError("Table interpolation error exceeds 1e-6; increase points.")
    record = {
        "alpha": curve.alpha,
        "boundary": boundary,
        "epsilon": curve.epsilon,
        "seeds": curve.seeds,
        "points": len(f),
        "plateau_f": calibration.plateau_f,
        "minimum_plateau_f": curve.minimum_plateau_f,
        "property_c_endpoint": curve.property_c_endpoint,
        "table_relative_error_on_midpoints": relative_error,
        "diagnostics": calibration.diagnostics.to_dict(),
        "reference": "https://doi.org/10.1257/aer.20211063",
        "algorithm": (
            "Appendix B.3; regularized inverse map; Property C; finite size search"
        ),
    }
    table = pa.table(
        [np.full(len(f), curve.alpha), f, regularized], schema=TABLE_SCHEMA
    )
    metadata = {repr(curve.alpha): record}
    if directory.joinpath(_MANIFEST_FILE).exists():
        existing, metadata = _read_dataset(directory)
        existing = existing.filter(
            pa.array(existing.column("alpha").to_numpy() != curve.alpha)
        )
        table = pa.concat_tables([existing.replace_schema_metadata(None), table])
        metadata[repr(curve.alpha)] = record
    table = table.sort_by([("alpha", "ascending"), ("f_stat", "ascending")])
    _from_dataset(curve.alpha, table, metadata)
    _write_dataset(table, metadata, directory)
    if include_csv:
        path = directory / f"alpha-{curve.alpha!r}.csv.gz"
        with (
            gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream,
        ):
            np.savetxt(
                stream,
                np.column_stack([f, calibration.critical_value(f), regularized]),
                delimiter=",",
                header="F,critical_value,branch_regularized_squared",
                comments="",
                fmt="%.17g",
            )
    return record


def load_table(
    alpha: float = 0.05, *, directory: str | Path | None = None
) -> Calibration:
    """Load a validated calibration, caching bundled tables by significance level.

    Args:
        alpha: Two-sided significance level with a precomputed table.
        directory: Custom dataset directory, read and validated on every call.

    Returns:
        Calibration reconstructed from the stored curve and diagnostics.

    Raises:
        ValueError: If the table is missing, corrupted, or inconsistent.
    """
    alpha = validate_alpha(alpha)
    if directory is None:
        return _load_bundled_table(alpha)
    root = Path(directory)
    if not root.joinpath(_MANIFEST_FILE).exists():
        raise ValueError(
            f"No table for alpha={alpha}; use calibrate(alpha) or method='compute'."
        )
    return _from_dataset(alpha, *_read_dataset(root))


@lru_cache(maxsize=32)
def _load_bundled_table(alpha: float) -> Calibration:
    return _from_dataset(alpha, *_bundled_dataset())
