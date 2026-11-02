import gzip
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tf_inference import (
    available_alphas,
    confidence_interval,
    construct_curve,
    critical_value,
    load_table,
)
from tf_inference.cli import main


def test_bundled_levels_and_intervals():
    assert len(available_alphas()) == 18
    for alpha in available_alphas():
        calibration = load_table(alpha)
        values = critical_value(
            [calibration.boundary, calibration.boundary + 0.01, 1e20], alpha
        )
        assert values[0] == np.inf
        assert np.isfinite(values[1:]).all()
        assert_allclose(values[-1], calibration.plateau)
        interval = confidence_interval(1, 0.4, calibration.boundary + 0.01, alpha)
        assert interval.shape == (2,)
        assert interval[0] < 1 < interval[1]


@pytest.mark.parametrize(
    "alpha,filename", [(0.05, "table3a.csv"), (0.01, "table3b.csv")]
)
def test_published_reference_values(alpha, filename):
    table = np.loadtxt(
        Path(__file__).parent / "references" / filename, delimiter=",", skiprows=1
    )
    f, printed = table.T
    curve = construct_curve(alpha)
    f_rounding = np.full(len(f), 0.001)
    if alpha == 0.05:
        f_rounding[-1] = 0.01
    actual = curve.critical(f)
    assert np.all(actual <= printed + 0.0001)
    assert np.all(curve.critical(f - f_rounding) >= printed - 0.0011)


def test_csv_values_match_public_api():
    from importlib.resources import files

    path = files("tf_inference").joinpath("data", "alpha-0.05.csv.gz")
    with path.open("rb") as stream, gzip.open(stream, "rt") as decoded:
        values = np.loadtxt(decoded, delimiter=",", skiprows=1)
    assert_allclose(critical_value(values[:, 0]), values[:, 1], rtol=1e-13)


def test_command_line(capsys):
    main(["critical", "3.8", "3.9", "10"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "F,critical_value"
    assert lines[1].endswith(",inf")
    assert_allclose(float(lines[-1].split(",")[1]), 3.4303537, atol=1e-8)


def test_compute_matches_table():
    assert_allclose(
        critical_value([3.9, 10, 1000], method="compute"),
        critical_value([3.9, 10, 1000], method="table"),
        rtol=1e-6,
    )


def test_no_alpha_interpolation():
    with pytest.raises(ValueError, match="No table"):
        critical_value(10, alpha=0.037, method="table")


def test_unknown_method():
    with pytest.raises(ValueError, match="method"):
        critical_value(10, method="unknown")


def test_command_line_generation(tmp_path, capsys):
    main(["generate", "--alpha", "0.05", "--output", str(tmp_path)])
    assert '"alpha": 0.05' in capsys.readouterr().out
    assert_allclose(
        load_table(0.05, directory=tmp_path).critical_value(10), 3.4303537, atol=1e-8
    )
