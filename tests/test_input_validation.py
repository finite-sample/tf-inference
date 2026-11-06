"""Public inference rejects non-real statistical inputs."""

import numpy as np
import pytest

from tf_inference import confidence_interval, critical_value, load_table


@pytest.mark.parametrize(
    "value", [10 + 5j, np.complex128(10 + 5j), np.array([10 + 5j])]
)
def test_nonreal_f_rejected(value):
    with pytest.raises(ValueError, match="real"):
        critical_value(value)
    with pytest.raises(ValueError, match="real"):
        load_table().curve.critical(value)


@pytest.mark.parametrize("argument", ["coefficient", "se", "f_stat"])
def test_nonreal_interval_inputs_rejected(argument):
    inputs = {"coefficient": 1, "se": 0.4, "f_stat": 10}
    inputs[argument] = np.array([1 + 2j])
    with pytest.raises(ValueError, match="real"):
        confidence_interval(**inputs)


@pytest.mark.parametrize("operation", [critical_value, load_table])
def test_nonreal_alpha_rejected(operation):
    args = (10,) if operation is critical_value else ()
    with pytest.raises(ValueError, match="real"):
        operation(*args, alpha=np.complex128(0.05 + 0.01j))


@pytest.mark.parametrize("argument", ["correlations", "strengths"])
def test_nonreal_calibration_grid_rejected(argument):
    from tf_inference import calibrate

    with pytest.raises(ValueError, match="real"):
        calibrate(**{argument: np.array([1 + 2j])})


@pytest.mark.parametrize("argument", ["strength", "rho", "plateau_f", "atol"])
def test_nonreal_probability_inputs_rejected(argument):
    from tf_inference import rejection_probability

    inputs = {"strength": 4.0, "rho": 0.9, "plateau_f": 100.0, "atol": 1e-9}
    inputs[argument] = np.complex128(1 + 2j)
    with pytest.raises(ValueError, match="real"):
        rejection_probability(load_table().curve, **inputs)
