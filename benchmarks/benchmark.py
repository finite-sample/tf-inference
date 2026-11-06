"""Measure lookup and fresh calibration separately, without timing imports."""

import argparse
import json
import platform
import statistics
import timeit
from pathlib import Path

import numpy as np
import scipy

import tf_inference as tf
from tf_inference import tables


def measure(function, *, number, repeat):
    """Measure per-call wall time over repeated samples."""
    samples = [
        value / number
        for value in timeit.repeat(function, number=number, repeat=repeat)
    ]
    return {
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "calls_per_sample": number,
    }


def main():
    """Write reproducible lookup and calibration timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.repeat < 3:
        parser.error("--repeat must be at least 3")

    def cold_load():
        tables._bundled_dataset.cache_clear()
        tables._load_bundled_table.cache_clear()
        return tf.load_table()

    results = {"cold_table_load": measure(cold_load, number=1, repeat=args.repeat)}
    calibration = tf.load_table()
    tf.critical_value(10)
    batch = np.linspace(0, 200, 100000)
    cases = {
        "warm_scalar": (lambda: tf.critical_value(10), 10000),
        "reused_scalar": (lambda: calibration.critical_value(10), 10000),
        "warm_array_100000": (lambda: tf.critical_value(batch), 100),
        "fresh_calibrate_0.05": (lambda: tf.calibrate(0.05), 1),
        "fresh_calibrate_0.037": (lambda: tf.calibrate(0.037), 1),
    }
    for name, (function, number) in cases.items():
        results[name] = measure(function, number=number, repeat=args.repeat)
        print(f"{name}: {results[name]['median_seconds']:.8g} seconds", flush=True)
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "notes": (
            "Imports excluded; cold load clears only the package cache, not OS caches. "
            "Calibration calls are uncached. Timings are local observations, "
            "not CI assertions."
        ),
        "benchmarks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
