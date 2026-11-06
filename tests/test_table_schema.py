"""Stored calibrations must satisfy their schema and numerical invariants."""

import hashlib
import json
from importlib.resources import files

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_allclose

from tf_inference import export_table, load_table


@pytest.fixture
def dataset(tmp_path):
    root = files("tf_inference").joinpath("data")
    for name in ("calibrations.parquet", "calibrations.json"):
        (tmp_path / name).write_bytes(root.joinpath(name).read_bytes())
    return tmp_path


def write_table(directory, table):
    path = directory / "calibrations.parquet"
    pq.write_table(table, path)
    manifest = {
        "schema_version": 2,
        "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (directory / "calibrations.json").write_text(json.dumps(manifest))


def test_arrow_schema_and_export_preserves_levels(tmp_path):
    export_table(load_table(0.05), tmp_path)
    export_table(load_table(0.01), tmp_path)
    table = pq.read_table(tmp_path / "calibrations.parquet")
    assert table.schema.names == ["alpha", "f_stat", "regularized_squared"]
    assert all(
        field.type == pa.float64() and not field.nullable for field in table.schema
    )
    assert set(table.column("alpha").to_pylist()) == {0.01, 0.05}
    assert not list(tmp_path.glob("*.csv.gz"))
    for alpha in (0.01, 0.05):
        expected = load_table(alpha).critical_value([4, 10, 100])
        actual = load_table(alpha, directory=tmp_path).critical_value([4, 10, 100])
        assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "corruption", ["dtype", "null", "nonfinite", "order", "alpha", "missing_metadata"]
)
def test_reject_invalid_parquet(dataset, corruption):
    table = pq.read_table(dataset / "calibrations.parquet")
    if corruption == "dtype":
        table = table.set_column(
            1,
            pa.field("f_stat", pa.float32(), nullable=False),
            table.column(1).cast(pa.float32()),
        )
    elif corruption == "missing_metadata":
        table = table.replace_schema_metadata(None)
    else:
        index = 0 if corruption == "alpha" else 1
        values = table.column(index).to_pylist()
        if corruption == "null":
            values[0] = None
        elif corruption == "nonfinite":
            values[0] = np.nan
        elif corruption == "order":
            values[1] = values[0]
        else:
            values[0] = 0.037
        field = table.schema.field(index)
        # Parquet writers reject nulls in non-nullable columns before loading.
        if corruption == "null":
            field = pa.field(field.name, field.type, nullable=True)
        table = table.set_column(index, field, pa.array(values, type=pa.float64()))
    write_table(dataset, table)
    with pytest.raises(ValueError, match=r"schema|metadata|finite|invariants"):
        load_table(0.05, directory=dataset)


@pytest.mark.parametrize(
    "corruption", ["boundary", "missing_field", "nan", "diagnostics", "duplicate_alpha"]
)
def test_reject_invalid_metadata(dataset, corruption):
    table = pq.read_table(dataset / "calibrations.parquet")
    metadata = json.loads(table.schema.metadata[b"tf_inference"])
    if corruption == "boundary":
        metadata["0.05"]["boundary"] = 4.0
    elif corruption == "missing_field":
        del metadata["0.05"]["seeds"]
    elif corruption == "nan":
        metadata["0.05"]["epsilon"] = float("nan")
    elif corruption == "diagnostics":
        metadata["0.05"]["diagnostics"]["plateau_f"] = 5.0
    else:
        metadata["0.050"] = metadata["0.05"]
    table = table.replace_schema_metadata(
        {b"tf_inference": json.dumps(metadata).encode()}
    )
    write_table(dataset, table)
    with pytest.raises(ValueError, match=r"metadata|invariants|diagnostics"):
        load_table(0.05, directory=dataset)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_rejection", float("nan")),
        ("integration_error", -1),
        ("worst_rho", 2),
        ("evaluations", True),
        ("strengths", []),
        ("correlations", [float("inf")]),
        ("strengths", [1000]),
    ],
)
def test_reject_invalid_diagnostics(dataset, field, value):
    table = pq.read_table(dataset / "calibrations.parquet")
    metadata = json.loads(table.schema.metadata[b"tf_inference"])
    metadata["0.05"]["diagnostics"][field] = value
    table = table.replace_schema_metadata(
        {b"tf_inference": json.dumps(metadata).encode()}
    )
    write_table(dataset, table)
    with pytest.raises(ValueError, match="diagnostics"):
        load_table(0.05, directory=dataset)


def test_platform_quantile_roundoff_uses_stored_boundary(dataset, monkeypatch):
    from tf_inference import tables

    original = tables.ndtri
    monkeypatch.setattr(tables, "ndtri", lambda x: np.nextafter(original(x), np.inf))
    actual = load_table(0.05, directory=dataset)
    expected = load_table(0.05)
    assert actual.boundary == expected.boundary
    f = np.nextafter(expected.boundary, np.inf)
    assert actual.critical_value(f) == expected.critical_value(f)


@pytest.mark.parametrize(
    ("field", "value"), [("maximum_rejection", 0.9), ("integration_error", 0.5)]
)
def test_diagnostics_must_describe_successful_calibration(dataset, field, value):
    table = pq.read_table(dataset / "calibrations.parquet")
    metadata = json.loads(table.schema.metadata[b"tf_inference"])
    metadata["0.05"]["diagnostics"][field] = value
    table = table.replace_schema_metadata(
        {b"tf_inference": json.dumps(metadata).encode()}
    )
    write_table(dataset, table)
    with pytest.raises(ValueError, match="diagnostics"):
        load_table(0.05, directory=dataset)


def test_decreasing_branch_required(dataset):
    table = pq.read_table(dataset / "calibrations.parquet")
    values = table.column("regularized_squared").to_numpy().copy()
    positions = np.flatnonzero(
        (table.column("alpha").to_numpy() == 0.05)
        & (table.column("f_stat").to_numpy() > 9)
    )
    values[positions[0]] *= 100
    table = table.set_column(2, table.schema.field(2), pa.array(values))
    write_table(dataset, table)
    with pytest.raises(ValueError, match="decreasing"):
        load_table(0.05, directory=dataset)


def test_decreasing_knots_do_not_imply_decreasing_interpolation(dataset):
    table = pq.read_table(dataset / "calibrations.parquet")
    metadata = json.loads(table.schema.metadata[b"tf_inference"])
    record = metadata["0.05"]
    f = np.array(
        [
            record["boundary"],
            3.8979517962086194,
            21.591309166952364,
            50.545915685432895,
            59.37927824729297,
            record["plateau_f"],
            record["property_c_endpoint"],
        ]
    )
    z = np.array(
        [
            record["boundary"] ** 3,
            15.495615727471716,
            4492.574358311533,
            1903.632819864526,
            347.07479468829695,
            387.3315714822953,
            454.21220828259146,
        ]
    )
    assert np.all(np.diff(np.sqrt(z[1:] / (f[1:] - f[0]))) < 0)
    record["points"] = len(f)
    table = pa.table([np.full(len(f), 0.05), f, z], schema=table.schema)
    table = table.replace_schema_metadata(
        {b"tf_inference": json.dumps({"0.05": record}).encode()}
    )
    write_table(dataset, table)
    with pytest.raises(ValueError, match="decreasing"):
        load_table(0.05, directory=dataset)
