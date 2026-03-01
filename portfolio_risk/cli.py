"""
Command-line interface for the portfolio risk analysis tool.

This is the "imperative shell" — the thinnest possible wrapper around
the pure pipeline. It handles:
    1. Parsing CLI arguments (argparse)
    2. Loading config from JSON file or CLI args
    3. Auto-detecting assets from the CSV
    4. Calling run_pipeline()
    5. Printing the result as formatted JSON

No business logic lives here. All computation happens in the pure core
(metrics.py, validators.py, pipeline.py).

Usage (CLI args):
    python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1
    python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 ... --risk-free-rate 0.02

Usage (config file):
    python -m portfolio_risk --config portfolio.json

Config file format:
    {
        "csv": "sample_returns.csv",
        "weights": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "risk_free_rate": 0.02
    }
"""

from __future__ import annotations

import argparse
import json
import sys

from portfolio_risk.pipeline import load_csv, detect_assets, run_pipeline


def load_config(config_path: str) -> dict | None:
    """
    Load a JSON config file.

    Pure-ish function: reads a file (I/O), but returns a plain dict
    or None on failure. No exceptions leak out.

    Args:
        config_path: Path to JSON config file.

    Returns:
        Dict with keys 'csv', 'weights', and optionally 'risk_free_rate'.
        None if file cannot be read or parsed.
    """
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the CLI.

    Supports two modes:
        1. --config portfolio.json (all settings in one file)
        2. --csv + --weights + optional --risk-free-rate (CLI args)

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
            "Examples:\n"
            "  # Using CLI arguments:\n"
            "  python -m portfolio_risk --csv sample_returns.csv "
            "--weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1\n\n"
            "  # Using a config file:\n"
            "  python -m portfolio_risk --config portfolio.json\n\n"
            "The tool auto-detects asset columns from the CSV. "
            "Weights are assigned to assets in column order."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        help="Path to JSON config file (alternative to --csv and --weights)",
    )

    parser.add_argument(
        "--csv",
        help="Path to CSV file with daily returns (columns: date, asset1, asset2, ...)",
    )

    parser.add_argument(
        "--weights",
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

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full results as JSON instead of human-readable summary",
    )

    return parser


def _print_error_and_exit(message: str) -> None:
    """Print a JSON error message and exit with code 1."""
    print(json.dumps({"status": "error", "message": message}, indent=2))
    sys.exit(1)


def _print_summary(result: dict) -> None:
    """
    Print a human-readable summary of the pipeline result.

    Shows key portfolio metrics as formatted text instead of raw JSON.
    Correlation matrix is compressed to top 3 positive and top 3 negative
    pairs — nobody reads a 10x10 matrix in a terminal.

    Args:
        result: The dict returned by run_pipeline().
    """
    if result["status"] == "error":
        print(f"Error: {result['message']}")
        return

    metrics = result["metrics"]
    config = result["config"]
    asset_names = config["asset_names"]
    volatility = metrics["portfolio_variance"] ** 0.5

    # Header
    print()
    print("Portfolio Risk Analysis")
    print("=" * 45)

    # Key metrics
    print(f"  Annualized Return:    {metrics['annualized_return']:>8.2%}")
    print(f"  Portfolio Volatility: {volatility:>8.2%}")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>8.4f}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:>8.4f}")
    print(f"  Max Drawdown:         {metrics['max_drawdown']:>8.2%}")
    print(f"  Win Rate:             {metrics['win_rate']:>8.2%}")
    print(f"  Risk-Free Rate:       {config['risk_free_rate']:>8.2%}")
    print()

    # Asset volatilities
    print("Asset Volatilities")
    print("-" * 45)
    vols = metrics["asset_volatilities"]
    for name, vol in zip(asset_names, vols):
        print(f"  {name:<12} {vol:>8.2%}")
    print()

    # Top correlations — extract upper triangle, sort, show top/bottom 3
    corr = metrics["correlation_matrix"]
    n = len(asset_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((asset_names[i], asset_names[j], corr[i][j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    top_n = min(3, len(pairs))

    if len(pairs) > 0:
        print("Top Positive Correlations")
        print("-" * 45)
        for a, b, c in pairs[:top_n]:
            print(f"  {a} / {b:<12} {c:>+.4f}")
        print()

        print("Top Negative Correlations")
        print("-" * 45)
        for a, b, c in pairs[-top_n:]:
            print(f"  {a} / {b:<12} {c:>+.4f}")
        print()

    # Warnings
    if result.get("warnings"):
        print("Warnings")
        print("-" * 45)
        print(f"  {result['warnings']}")
        print()


def main() -> None:
    """
    Main entry point for the CLI.

    Flow:
        1. Parse arguments (config file or CLI args)
        2. Load CSV to auto-detect assets
        3. Validate weight count against detected assets (friendly error message)
        4. Run pipeline
        5. Print JSON result
    """
    parser = build_parser()
    args = parser.parse_args()
    use_json = args.json

    # Step 1: Resolve config — either from JSON file or CLI args
    if args.config:
        config = load_config(args.config)
        if config is None:
            _print_error_and_exit(f"Could not read or parse config file: {args.config}")

        csv_path = config.get("csv")
        weights_list = config.get("weights")
        risk_free_rate = config.get("risk_free_rate", 0.0)

        if csv_path is None or weights_list is None:
            _print_error_and_exit(
                "Config file must contain 'csv' and 'weights' keys. "
                "Example: {\"csv\": \"data.csv\", \"weights\": [0.5, 0.5]}"
            )

        weights = tuple(weights_list)
    elif args.csv and args.weights:
        csv_path = args.csv
        weights = tuple(args.weights)
        risk_free_rate = args.risk_free_rate
    else:
        _print_error_and_exit(
            "Provide either --config <file.json> or both --csv and --weights. "
            "Run with --help for usage."
        )
        return  # unreachable, but makes type checker happy

    # Step 2: Load CSV to auto-detect assets
    raw_df = load_csv(csv_path)
    if raw_df is None:
        _print_error_and_exit(f"Could not read CSV file: {csv_path}")

    # Step 3: Auto-detect asset columns
    detected_assets = detect_assets(raw_df)
    if len(detected_assets) == 0:
        _print_error_and_exit("No numeric asset columns found in CSV.")

    # Step 4: Check weight count with a friendly message
    if len(weights) != len(detected_assets):
        asset_list = ", ".join(detected_assets)
        _print_error_and_exit(
            f"Found {len(detected_assets)} assets in CSV "
            f"({asset_list}) but received {len(weights)} weights. "
            f"Please provide exactly {len(detected_assets)} weights."
        )

    # Step 5: Run the pipeline
    result = run_pipeline(
        file_path=csv_path,
        weights=weights,
        asset_names=detected_assets,
        risk_free_rate=risk_free_rate,
    )

    # Step 6: Print result
    if use_json:
        print(json.dumps(result, indent=2))
    else:
        _print_summary(result)

    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
