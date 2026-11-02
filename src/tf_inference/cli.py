"""Command-line evaluation and reproducible table generation."""

import argparse
import json

import numpy as np

from . import calibrate, critical_value, export_table


def main(argv=None):
    """Evaluate critical values or generate portable calibration files."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("critical", help="Evaluate tF critical values")
    evaluate.add_argument("F", nargs="+", type=float)
    evaluate.add_argument("--alpha", type=float, default=0.05)
    evaluate.add_argument("--method", choices=["table", "compute"], default="table")
    generate = commands.add_parser(
        "generate", help="Construct and export numerical tables"
    )
    generate.add_argument("--alpha", nargs="+", type=float, required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--points", type=int, default=16385)
    generate.add_argument("--epsilon", type=float, default=1e-4)
    generate.add_argument("--seeds", type=int, default=256)
    generate.add_argument("--size-tolerance", type=float, default=2e-7)
    generate.add_argument("--strength-limit", type=float, default=80)
    args = parser.parse_args(argv)
    if args.command == "critical":
        try:
            values = np.atleast_1d(
                critical_value(args.F, args.alpha, method=args.method)
            )
        except ValueError as error:
            parser.error(str(error))
        print("F,critical_value")
        for f_stat, value in zip(args.F, values, strict=True):
            print(f"{f_stat:.17g},{value:.17g}")
    else:
        for alpha in args.alpha:
            calibration = calibrate(
                alpha,
                epsilon=args.epsilon,
                seeds=args.seeds,
                size_tolerance=args.size_tolerance,
                strength_limit=args.strength_limit,
            )
            metadata = export_table(calibration, args.output, points=args.points)
            print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
