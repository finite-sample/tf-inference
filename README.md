# tf-inference

![PyPI](https://img.shields.io/pypi/v/tf-inference.svg)
[![CI](https://github.com/finite-sample/tf-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/tf-inference/actions/workflows/ci.yml)
![Python](https://img.shields.io/pypi/pyversions/tf-inference.svg)

tf-inference computes tF critical values and confidence intervals for
instrumental-variable (IV) models with one endogenous regressor and one
excluded instrument. It provides generated tables for common confidence
levels and numerical calibration for other levels.

The tF method and its theoretical results are due to David S. Lee,
Justin McCrary, Marcelo J. Moreira and Jack Porter, authors of [Valid
t-Ratio Inference for IV (2022)](https://doi.org/10.1257/aer.20211063).
This package provides an independent Python implementation of their
Appendix B.3 construction and reusable numerical tables. Critical values
depend on the first-stage F statistic and a two-sided significance
level. The code constructs the curve from its boundary expansion and
recurrence; the published rounded tables serve as test references.

This is an initial numerical implementation. Its calibration uses
deterministic integration, a finite nuisance-parameter grid and local
searches. These checks are documented and reproducible; they are not a
proof of a uniform size bound. The inferential framework is asymptotic
under the single-instrument IV model, not a finite-sample coverage
guarantee or a remedy for invalid instruments.

## Installation

Install with Python 3.12 or later:

    python -m pip install tf-inference

NumPy, SciPy and PyArrow are required at runtime. No Stata, external data
download, API key or network access is needed to evaluate or generate
critical values.

## Use the generated tables

``` python
from tf_inference import critical_value, confidence_interval

critical_value([3.8, 3.9, 10, 1000], alpha=0.05)
# Approximately [inf, 30.93, 3.43035, 1.95996]

confidence_interval(coefficient=1, se=0.4, f_stat=10, alpha=0.05)
# Approximately [-0.37214, 2.37214]
```

`alpha` is the significance level: 0.05 means 95% confidence. The return
value of `critical_value` is the threshold for **absolute t**, not t
squared or a standard-error multiplier. Divide it by the normal
two-sided critical value to obtain the corresponding standard-error
multiplier for that alpha.

Inputs broadcast as NumPy arrays. Confidence intervals put lower and
upper bounds on the final axis. Complex inputs, nonfinite or negative F statistics,
nonfinite coefficients, and nonpositive standard errors are rejected.

The 95% boundary is about 3.84145882. F at or below the boundary returns
an infinite critical value; F strictly above it has a finite critical
value. There is no artificial cutoff at F=4. The boundary is computed
separately for each alpha as `scipy.special.ndtri(alpha / 2)**2`. Very
close to the boundary, rounding in the input F itself can materially
affect the result.

Bundled tables cover alpha values 0.0001, 0.0005, 0.001, 0.002, 0.005,
0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1, 0.125, 0.15, 0.2
and 0.25. Each contains at least 16,385 rows, concentrated near the
boundary. List the installed levels with `available_alphas()`. Tables
are never interpolated across alpha values.

The Python convenience functions and CLI default to `method="table"`. A
missing level raises an error; numerical calibration requires an
explicit `method="compute"` or a direct call to `calibrate`. The `auto`
method is not supported.

## Calculate an unconventional level

``` python
from tf_inference import calibrate

calibration = calibrate(alpha=0.037)
calibration.critical_value([4.5, 10, 100])
calibration.confidence_interval(1, 0.4, 10)
print(calibration.diagnostics)
```

The bundled Parquet dataset is read and validated once; interpolators are
cached per alpha and shared by `load_table`,
`critical_value` and `confidence_interval`. Repeated lookup does not
read files, verify checksums again or rebuild the interpolator. Explicit
`method="compute"` calls use a separate cache of up to 32 calibrations.
Direct `calibrate` calls always compute afresh. Tables loaded from a
custom directory are read and checksum-verified on each call so changes
are detected.

For large inputs, pass a NumPy array in one call. For repeated work at
one level, retain `load_table(alpha)` or the result of
`calibrate(alpha)` and call its methods directly.

The numerical alpha domain is \[0.0001, 0.25\]. It covers confidence
levels from 75% to 99.99%. This is an explicit implementation range, not
a theoretical limit of tF. Calculations outside it are rejected rather
than extrapolated.

`calibrate` accepts `epsilon`, `seeds`, `size_tolerance`,
`strength_limit`, `strengths` and `correlations`. Smaller initialization
epsilon and more seeds allow convergence checks. These settings are not
claims of certified error. Failed recurrence, integration or
optimization raises `ConvergenceError`. Inspect the returned diagnostics
when doing new calibrations, and validate beyond the search settings
before relying on an unconventional level for research.

## Generate constants for other software

The command-line interface exposes the same calculation:

    tf-inference critical 3.9 10 100 --alpha 0.05
    tf-inference critical 10 --alpha 0.037 --method compute
    tf-inference generate --alpha 0.05 0.01 0.037 --output tables

The generator writes one `calibrations.parquet` dataset and a
`calibrations.json` manifest. Adding or replacing one alpha preserves the other
levels in that directory. The Parquet schema has three nonnullable float64
columns: `alpha`, `f_stat` and `regularized_squared`. Calibration settings,
plateaus and diagnostics are embedded as JSON in the Arrow schema metadata.
The schema-versioned manifest records the dataset's SHA-256 checksum.
Loading validates the checksum, schema, finite values, ordered grids,
boundary and plateau invariants, and agreement with the metadata.

`export_table(calibration, directory, points=16385)` and
`load_table(alpha, directory=directory)` provide the same operations in Python.
For a portable gzip CSV, add `--csv` to the generation command or pass
`include_csv=True` to `export_table`. CSV files are optional exports, not
packaged runtime data.

For accurate boundary evaluation, interpolate the regularized quantity
`Z(F) = (F - boundary) * critical(F)**2` using a monotone cubic interpolator.
It is the Parquet `regularized_squared` column (and the optional CSV's
`branch_regularized_squared` column). Then compute
`sqrt(Z(F) / (F - boundary))`. Cap F at the metadata's `plateau_f` before
evaluation and return infinity when F is at or below the boundary. The raw
branch extends to the Property (C) endpoint; at large F use the selected
plateau. The CSV's first critical value is infinite and must not be
interpolated as an ordinary finite row.

The export measures relative interpolation differences at every cell
midpoint against the computed curve, and rejects tables exceeding 1e-6
there. This is a measured check, not an interval-arithmetic error bound.
The shipped validation also compares randomly chosen F values against a
separately refined curve.

## How the calculation works

1.  Compute the boundary q and initialize the squared critical value
    near it using the expansion in Appendix B.3, step 1.
2.  Apply the inverse map in step 2 to extend the decreasing curve. The
    code interpolates Z(F), whose finite boundary limit is q cubed,
    rather than interpolating a divergent critical value.
3.  Locate the tangency that limits the decreasing segment under
    Property (C). Its admissible plateau is also constrained to be at
    least the normal two-sided critical value.
4.  Integrate null rejection probabilities in the Gaussian weak-IV
    experiment. Search the nuisance parameters and move the plateau
    earlier if it over-rejects beyond the requested numerical tolerance.

The size search can change the plateau. For example, alpha=0.037
requires an adjustment to the first candidate. For alpha=0.01 the
plateau is about 2.72549, above the normal 99% critical value of about
2.57583. Replacing the 95% normal quantile with another quantile does
not implement general tF.

The default size search covers strength f0 from 0 to 80 and absolute
correlation rho from 0 to 1. Here E\[F\] = 1 + f0 squared; f0 is not F.
It includes singular rho=1 calculations, correlations approaching one,
and local searches between grid points. The strong-instrument limit is
checked separately by the normal critical-value constraint. The
unbounded nuisance domain is not exhaustively certified. The recorded
integration error does not include curve approximation or missed extrema
in the parameter search.

## Performance

Table lookup avoids numerical calibration. For large inputs, evaluate an array
in one call. `make benchmark` records separate timings for warm scalar and
array lookup, loading the complete Parquet dataset with empty package caches,
and fresh calibration. Imports and operating-system file-cache effects are
excluded. Results and environment details are in `benchmarks/results.json`;
`benchmarks/baseline.json` preserves the earlier implementation's measurements.
These are local observations, not CI timing assertions. Tests enforce that
default calls never calibrate and warm calls avoid file access and interpolator
construction.

## Validation and development

Install development dependencies and run the checks:

    uv sync --all-groups
    make ci

Tests cover the published rounded-table values and plateau landmarks,
initialization and grid refinement, the defining two-intersection
equations, monotonicity and the exact boundary, near-singular numerical
integration, independent Gaussian Monte Carlo calculations,
unconventional-alpha plateau adjustment, interval broadcasting, table
roundtrips and corrupted files, Parquet schema violations, complex inputs and
large-strength numerical stability. `make ci` runs Ruff, Pyright, pydoclint,
tests with a 90% coverage floor (including branch measurement), documentation
and distribution checks.
`preen check --strict` checks adoption of the shared py-canon standard.

`validation/validate.py` evaluates bundled tables against finer
recurrence calculations and held-out nuisance-parameter grids extending
to f0=160. `validation/results.json` records all cases and the
environment. Reproduce it:

    uv run python validation/validate.py --output validation/results.json

Rebuild the bundled constants from scratch:

    make tables

`make ci-docker` uses the standard `python:3.12` image. Documentation
includes this README directly; it is not maintained as a second copy.

## Scope and integration

This package takes summary quantities. It does not fit an IV model or
verify that an instrument is valid. For use with an estimator, require
one endogenous regressor and one excluded instrument, and use the
appropriate first-stage squared t statistic and compatible covariance
estimates. It supplies the numerical calibration; estimator-specific
eligibility checks belong in the calling package.

This implements the 2022 tF construction. VtF is a different procedure
using additional information and is outside the present implementation.
No generic p-value API is provided: numerical nesting across arbitrary
alpha levels would need separate validation before inverting the family
of tests.

## Citation and sources

Please cite the paper when using this package for research:

Lee, David S., Justin McCrary, Marcelo J. Moreira, and Jack Porter.
2022. [Valid t-Ratio Inference for
IV](https://doi.org/10.1257/aer.20211063). *American Economic Review*
112 (10): 3260-3290.

``` bibtex
@article{lee2022valid,
  author = {Lee, David S. and McCrary, Justin and Moreira, Marcelo J. and Porter, Jack},
  title = {Valid {t}-Ratio Inference for {IV}},
  journal = {American Economic Review},
  year = {2022},
  volume = {112},
  number = {10},
  pages = {3260--3290},
  doi = {10.1257/aer.20211063}
}
```

The authors provide the construction and supporting materials:

- [Author manuscript with Appendix
  B.3](https://eml.berkeley.edu/~jmccrary/LMMP_2_15_22.pdf), February
  16, 2022.
- [Published supplemental
  appendix](https://www.aeaweb.org/articles/materials/17480).
- [Data and code for the paper](https://doi.org/10.3886/E164502V1), the
  authors' replication package.
- [Authors' supplementary
  material](https://irs.princeton.edu/davidlee-supplementarytF) and
  [Stata implementation](https://www.princeton.edu/~davidlee/wp/tf.ado).

AI assistance was used in this package's development and validation. The
numerical limitations and executed checks are recorded above and in the
repository's validation results.
