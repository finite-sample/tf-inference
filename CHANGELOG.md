# Changelog

## [Unreleased]

## 0.1.0 — 2026-11-06

First release of the independent Python implementation of the 2022 tF
construction for one endogenous regressor and one excluded instrument.

- Provide cached critical values and confidence intervals at 18 significance
  levels, with explicit numerical calibration for other supported levels.
- Store generated curves in one typed Parquet dataset with a schema-versioned
  checksum manifest, calibration diagnostics and validation on load.
- Offer optional CSV export for other statistical software.
- Reject complex inference inputs and stabilize rejection-probability
  integration at extremely large finite instrument strength.
- Adopt py-canon development, documentation and release workflows; require
  Python 3.12 or later, NumPy, SciPy and PyArrow.

Calibration checks a finite nuisance-parameter domain. It does not certify a
uniform size bound or finite-sample coverage. See the README for the numerical
method, validation and limitations.
