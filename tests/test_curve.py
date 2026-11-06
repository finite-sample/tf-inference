import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ndtr, ndtri

from tf_inference import construct_curve
from tf_inference.curve import inverse_map


@pytest.fixture(scope="module")
def curve():
    return construct_curve()


@pytest.mark.parametrize("alpha", [0.0001, 0.001, 0.01, 0.037, 0.05, 0.1, 0.25])
def test_seed_and_resolution_convergence(alpha):
    coarse = construct_curve(alpha)
    fine = construct_curve(alpha, epsilon=2.5e-5, seeds=512)
    end = min(coarse.minimum_plateau_f, fine.minimum_plateau_f)
    f = coarse.boundary + np.geomspace(1e-10, end - coarse.boundary, 2001)
    assert_allclose(coarse.critical(f), fine.critical(f), rtol=2e-6)
    assert_allclose(coarse.property_c_endpoint, fine.property_c_endpoint, rtol=2e-6)


def test_boundary_and_asymptotic(curve):
    q = ndtri(0.025) ** 2
    assert curve.boundary == q
    assert np.isinf(curve.critical(q))
    assert np.isinf(curve.critical(np.nextafter(q, 0)))
    f = np.nextafter(q, np.inf)
    assert np.isfinite(curve.critical(f))
    assert_allclose(curve.critical(f) ** 2 * (f - q), q**3, rtol=1e-12)
    assert_allclose(curve.critical(3.9), 30.945, rtol=0.002)


@pytest.mark.parametrize(
    ("f", "rounded"),
    [(4, 18.656), (5.002, 6.854), (9.835, 3.463), (24.605, 2.461), (104.67, 1.96)],
)
def test_published_95_table(curve, f, rounded):
    # Table 3a rounds BOTH F and the critical value upward to three decimals.
    assert curve.critical(f) <= rounded + 1e-4
    assert curve.critical(f - 0.001) >= rounded - 0.0011


def test_published_plateau_landmarks(curve):
    assert_allclose(curve.minimum_plateau_f, 104.67075, atol=0.0001)
    curve_99 = construct_curve(0.01)
    assert_allclose(curve_99.property_c_endpoint, 252.34266, atol=0.001)
    assert_allclose(curve_99.critical(curve_99.property_c_endpoint), 2.726, atol=0.001)


def test_two_intersection_equations(curve):
    f = np.linspace(3.85, 9, 60)
    c = curve.critical(f)
    next_f, next_c = inverse_map(f, c, curve.alpha)
    m = f / (c - np.sqrt(f))
    left, right = -np.sqrt(f), np.sqrt(next_f)
    assert_allclose(ndtr(right - m) - ndtr(left - m), 0.95, atol=2e-15)
    assert_allclose(left**2 * (left - m) ** 2 / m**2, c**2, rtol=2e-14)
    assert_allclose(right**2 * (right - m) ** 2 / m**2, next_c**2, rtol=2e-14)
    assert_allclose(curve.critical(next_f), next_c, rtol=2e-7)


def test_monotonicity(curve):
    f = curve.boundary + np.geomspace(1e-12, 1000, 20001)
    assert np.all(np.diff(curve.critical(f)) < 0)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, -np.inf])
def test_invalid_scalar_and_vector_f(curve, value):
    for argument in [value, np.array(value), [value]]:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            curve.critical(argument)


def test_scalar_and_vector_domain_limit(curve):
    value = np.nextafter(curve.f[-1], np.inf)
    for argument in [value, np.array(value), [value]]:
        with pytest.raises(ValueError, match="outside"):
            curve.critical(argument)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, np.nan, np.inf, 0.00001, 0.3])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        construct_curve(alpha)


@pytest.mark.parametrize(
    "kwargs", [{"epsilon": 0}, {"epsilon": np.nan}, {"seeds": 1}, {"seeds": 32.5}]
)
def test_invalid_numerical_settings(kwargs):
    with pytest.raises(ValueError, match="must be"):
        construct_curve(**kwargs)
