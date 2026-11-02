"""Held-out checks of bundled tables and their numerical construction.

Run: python validation/validate.py --output validation/results.json
This does not alter calibration settings after looking at held-out results.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.special import ndtr

from tf_inference import (
    available_alphas,
    construct_curve,
    load_table,
    rejection_probability,
)


def validate(alpha):
    table = load_table(alpha)
    reference = construct_curve(alpha, epsilon=2.5e-5, seeds=512)
    rng = np.random.default_rng(20261102)
    q = table.boundary
    f = q + np.exp(rng.uniform(np.log(1e-10), np.log(table.plateau_f - q), 20000))
    reference_critical = reference.critical(np.minimum(f, table.plateau_f))
    error = float(np.max(np.abs(table.critical_value(f) / reference_critical - 1)))
    monotone_f = q + np.geomspace(1e-12, 1e8, 40001)
    monotone = bool(np.all(np.diff(table.critical_value(monotone_f)) <= 0))
    stop = min(table.plateau_f, reference.property_c_endpoint)
    # These strengths and correlations are absent from the generation grid.
    strengths = np.r_[0.001, 0.015, 0.075, 0.33, np.geomspace(0.7, 160, 24)]
    correlations = [0.125, 0.6, 0.875, 0.97, 0.997, 0.9997, 0.99999, 1.0]
    records = []
    for m in strengths:
        for rho in correlations:
            result = rejection_probability(reference, float(m), rho, plateau_f=stop)
            records.append(
                {
                    "strength": float(m),
                    "rho": rho,
                    "rejection": result.probability,
                    "integration_error": result.error,
                }
            )
    maximum = max(records, key=lambda row: row["rejection"])
    limit = float(2 * ndtr(-table.plateau))
    passed = (
        error < 2e-6
        and monotone
        and maximum["rejection"] <= alpha + 5e-7
        and limit <= alpha + 1e-12
    )
    return {
        "alpha": alpha,
        "passed": passed,
        "max_relative_difference_to_refined_curve": error,
        "monotone": monotone,
        "strong_instrument_limit": limit,
        "worst_held_out_case": maximum,
        "held_out_cases": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="validation/results.json")
    args = parser.parse_args()
    results = []
    for alpha in available_alphas():
        result = validate(alpha)
        results.append(result)
        print(
            alpha,
            result["passed"],
            result["max_relative_difference_to_refined_curve"],
            result["worst_held_out_case"],
            flush=True,
        )
    output = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scope": (
            "Finite held-out numerical checks; "
            "not a uniform size proof or finite-sample guarantee."
        ),
        "refinement": {"epsilon": 2.5e-5, "seeds": 512},
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    if not results or not all(result["passed"] for result in results):
        raise SystemExit("Validation failed.")


if __name__ == "__main__":
    main()
