"""
Command-line interface for the portfolio risk analysis tool.

This is the "imperative shell" — the thinnest possible wrapper around
the pure pipeline. It handles:
    1. Parsing CLI arguments (argparse)
    2. Auto-detecting assets from the CSV
    3. Calling run_pipeline()
    4. Printing the result as formatted JSON

No business logic lives here. All computation happens in the pure core
(metrics.py, validators.py, pipeline.py).

Usage:
    python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
    python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 --risk-free-rate 0.02
"""

import argparse
import json
import sys

from portfolio_risk.pipeline import load_csv, detect_assets, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the CLI.

    Separated into its own function for testability — you can inspect
    the parser without actually running the CLI.
    """
    parser = argparse.ArgumentParser(
        prog="portfolio_risk",
        description=(
            "Analyze a portfolio of assets for basic risk metrics. "
            "Reads a CSV of daily returns, validates the data, and computes "
            "portfolio variance, Sharpe ratio, max drawdown, and more."
        ),
        epilog=(
            "Example:\n"
            "  python -m portfolio_risk --csv sample_returns.csv "
            "--weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1\n\n"
            "The tool auto-detects asset columns from the CSV. "
            "Weights are assigned to assets in column order."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file with daily returns (columns: date, asset1, asset2, ...)",
    )

    parser.add_argument(
        "--weights",
        required=True,
        nargs="+",
        type=float,
        help="Portfolio weights for each asset (must sum to 1.0, assigned in column order)",
    )

    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annualized risk-free rate for Sharpe ratio (default: 0.0)",
    )

    return parser


def main() -> None:
    """
    Main entry point for the CLI.

    Flow:
        1. Parse arguments
        2. Load CSV to auto-detect assets
        3. Validate weight count against detected assets (friendly error message)
        4. Run pipeline
        5. Print JSON result
    """
    parser = build_parser()
    args = parser.parse_args()

    # Step 1: Load CSV to auto-detect assets
    raw_df = load_csv(args.csv)
    if raw_df is None:
        print(json.dumps({
            "status": "error",
            "message": f"Could not read CSV file: {args.csv}",
        }, indent=2))
        sys.exit(1)

    # Step 2: Auto-detect asset columns
    detected_assets = detect_assets(raw_df)
    if len(detected_assets) == 0:
        print(json.dumps({
            "status": "error",
            "message": "No numeric asset columns found in CSV.",
        }, indent=2))
        sys.exit(1)

    # Step 3: Check weight count with a friendly message
    weights = tuple(args.weights)
    if len(weights) != len(detected_assets):
        asset_list = ", ".join(detected_assets)
        print(json.dumps({
            "status": "error",
            "message": (
                f"Found {len(detected_assets)} assets in CSV "
                f"({asset_list}) but received {len(weights)} weights. "
                f"Please provide exactly {len(detected_assets)} weights."
            ),
        }, indent=2))
        sys.exit(1)

    # Step 4: Run the pipeline
    result = run_pipeline(
        file_path=args.csv,
        weights=weights,
        asset_names=detected_assets,
        risk_free_rate=args.risk_free_rate,
    )

    # Step 5: Print JSON result
    print(json.dumps(result, indent=2))

    # Exit with appropriate code
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
