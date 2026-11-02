import numpy as np
import pytest
from numpy.testing import assert_allclose

from tf_inference import (
    calibrate,
    construct_curve,
    export_table,
    load_table,
    rejection_probability,
)


@pytest.fixture(scope="module")
def calibration():
    return calibrate()


def test_uncalibrated_unconventional_alpha_overrejects():
    curve = construct_curve(0.037)
    assert rejection_probability(curve, 16, 1).probability > 0.0371
    calibrated = calibrate(0.037)
    assert calibrated.diagnostics.adjustments > 0
    assert calibrated.plateau_f < curve.minimum_plateau_f
    # Hold out strengths and a correlation absent from the calibration grid.
    for m in [13.5, 15.7, 16.3, 19.1, 100]:
        for rho in [0.97, 0.9997, 1]:
            result = rejection_probability(
                curve, m, rho, plateau_f=calibrated.plateau_f
            )
            assert result.probability <= 0.037 + 3e-7


def test_calibration_diagnostics(calibration):
    assert_allclose(calibration.plateau, 1.95996398454, atol=1e-10)
    assert calibration.diagnostics.maximum_rejection < 0.0500002
    assert calibration.diagnostics.evaluations > 400


def test_interval_and_broadcasting(calibration):
    result = calibration.confidence_interval(1, 0.4, [3, 3.9, 10, 1000])
    assert result.shape == (4, 2)
    assert_allclose(result[0], [-np.inf, np.inf])
    assert np.isfinite(result[1:]).all()
    assert_allclose(result[2], [1 - 0.4 * 3.43035370, 1 + 0.4 * 3.43035370], atol=1e-8)
    assert_allclose(
        result[3], [1 - 0.4 * calibration.plateau, 1 + 0.4 * calibration.plateau]
    )


def test_table_roundtrip_and_checksum(calibration, tmp_path):
    metadata = export_table(calibration, tmp_path)
    restored = load_table(0.05, directory=tmp_path)
    f = np.r_[0, calibration.boundary, 3.9, np.geomspace(4, 1e6, 1000)]
    assert_allclose(
        restored.critical_value(f), calibration.critical_value(f), rtol=1e-6
    )
    assert restored.diagnostics == calibration.diagnostics
    assert metadata["table_relative_error_on_midpoints"] < 1e-6
    table_file = tmp_path / "alpha-0.05.npz"
    table_file.write_bytes(table_file.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_table(0.05, directory=tmp_path)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_invalid_f(calibration, value):
    with pytest.raises(ValueError, match="F"):
        calibration.critical_value(value)


@pytest.mark.parametrize("se", [0, -1, np.nan, np.inf])
def test_invalid_standard_error(calibration, se):
    with pytest.raises(ValueError, match="se"):
        calibration.confidence_interval(1, se, 10)


def test_missing_table(tmp_path):
    with pytest.raises(ValueError, match="No table"):
        load_table(0.037, directory=tmp_path)
