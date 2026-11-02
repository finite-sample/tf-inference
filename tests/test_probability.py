import numpy as np
import pytest
from numpy.testing import assert_allclose

from tf_inference import construct_curve, rejection_probability


@pytest.fixture(scope="module")
def curve():
    return construct_curve()


def test_singular_zero_strength(curve):
    assert rejection_probability(curve, 0, 1).probability == curve.alpha


@pytest.mark.parametrize("strength", [0.001, 0.1, 1, 4, 8])
def test_exact_boundary_size(curve, strength):
    result = rejection_probability(curve, strength, 1)
    assert_allclose(result.probability, 0.05, atol=2e-9)


@pytest.mark.parametrize("rho", [0.999, 0.9999, 0.99999])
def test_narrow_transitions_at_zero_strength(curve, rho):
    zero = rejection_probability(curve, 0, rho)
    small = rejection_probability(curve, 0.001, rho)
    assert abs(zero.probability - small.probability) < 1e-8
    assert zero.probability < 0.05
    assert zero.error < 1e-8


@pytest.mark.parametrize(
    "strength,rho", [(0, 0), (2, 0.5), (4, 0.9), (16, 0.99), (16, 1)]
)
def test_independent_gaussian_monte_carlo(curve, strength, rho):
    rng = np.random.default_rng(9133)
    count = 400000
    x, u = rng.standard_normal((2, count))
    f = strength + x
    ar = rho * x + np.sqrt(1 - rho**2) * u
    denominator = f**2 - 2 * rho * f * ar + ar**2
    critical = curve.critical(np.minimum(f**2, curve.minimum_plateau_f))
    rejected = ar**2 * f**2 > critical**2 * denominator
    rate = np.mean(rejected)
    expected = rejection_probability(curve, strength, rho).probability
    se = np.sqrt(max(expected * (1 - expected), 1 / count) / count)
    assert abs(rate - expected) < 5 * se + 1 / count


def test_sign_symmetry(curve):
    expected = rejection_probability(curve, 4, 0.9)
    assert rejection_probability(curve, -4, -0.9) == expected


@pytest.mark.parametrize("strength,rho", [(np.nan, 0), (0, np.nan), (1, 2)])
def test_invalid_inputs(curve, strength, rho):
    with pytest.raises(ValueError):
        rejection_probability(curve, strength, rho)
