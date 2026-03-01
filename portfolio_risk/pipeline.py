"""
Pipeline composition: chains all steps from CSV loading to metric output.

Architecture: "functional core, imperative shell"
    - load_csv() is the ONLY impure function (file I/O) — the "shell"
    - Everything after loading is pure function composition — the "core"

Pipeline flow:
    load_csv → validate_data → validate_weights → compute_all_metrics → to_dict

The main entry point is run_pipeline(), which returns a JSON-serializable dict
containing either the computed metrics or an error message. This keeps all
error handling explicit (no exceptions) and makes the output predictable.
"""

from __future__ import annotations

import polars as pl

from portfolio_risk.models import (
    DataValidationResult,
    PortfolioConfig,
    ReturnsData,
    RiskMetrics,
    WeightValidationResult,
)
from portfolio_risk.validators import validate_data, validate_weights
from portfolio_risk.metrics import (
    compute_portfolio_variance,
    compute_annualized_return,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_asset_volatilities,
    compute_correlation_matrix,
    compute_win_rate,
)


# ============================================================
# Impure Shell: File I/O
# ============================================================

def load_csv(file_path: str) -> pl.DataFrame | None:
    """
    Load a CSV file into a Polars DataFrame.

    This is the ONLY impure function in the pipeline — it performs file I/O.
    All other functions are pure (same inputs → same outputs, no side effects).

    Polars note: pl.read_csv() is the equivalent of pd.read_csv().
    Unlike pandas, the resulting DataFrame is immutable by default.

    Args:
        file_path: Path to the CSV file.

    Returns:
        Polars DataFrame if successful, None if file cannot be read.
    """
    try:
        return pl.read_csv(file_path)
    except Exception:
        return None


def detect_assets(df: pl.DataFrame) -> tuple[str, ...]:
    """
    Auto-detect asset columns from a DataFrame.

    Pure function: identifies numeric columns, excludes 'date' column.
    Returns asset names in the order they appear in the CSV.

    Args:
        df: Raw Polars DataFrame loaded from CSV.

    Returns:
        Tuple of asset column names (immutable).
    """
    return tuple(
        col for col in df.columns
        if col != "date" and df[col].dtype.is_numeric()
    )


# ============================================================
# Pure Core: Metric Computation
# ============================================================

def compute_all_metrics(
    returns_data: ReturnsData,
    config: PortfolioConfig,
) -> RiskMetrics:
    """
    Compute all risk metrics for the portfolio.

    Pure function: takes validated data and config, returns immutable RiskMetrics.
    Composes all individual metric functions into a single result.

    Args:
        returns_data: Validated and cleaned returns data.
        config: Validated portfolio configuration with weights.

    Returns:
        Frozen RiskMetrics dataclass containing all computed values.
    """
    returns_df = returns_data.data
    weights = config.weights

    return RiskMetrics(
        portfolio_variance=compute_portfolio_variance(returns_df, weights),
        annualized_return=compute_annualized_return(returns_df, weights),
        sharpe_ratio=compute_sharpe_ratio(returns_df, weights, config.risk_free_rate),
        sortino_ratio=compute_sortino_ratio(returns_df, weights, config.risk_free_rate),
        max_drawdown=compute_max_drawdown(returns_df, weights),
        win_rate=compute_win_rate(returns_df, weights),
        asset_volatilities=compute_asset_volatilities(returns_df),
        correlation_matrix=compute_correlation_matrix(returns_df),
    )


# ============================================================
# Pipeline Composition
# ============================================================

def run_pipeline(
    file_path: str,
    weights: tuple[float, ...],
    asset_names: tuple[str, ...],
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Run the full portfolio risk analysis pipeline.

    Chains: load → validate_data → validate_weights → compute_metrics → to_dict

    Each step checks the result of the previous step before proceeding.
    If any step fails, the pipeline returns an error dict immediately
    (no exceptions — explicit FP-style error handling).

    Args:
        file_path: Path to CSV file with daily returns.
        weights: User-provided portfolio weights (one per asset).
        asset_names: Asset names corresponding to the weights.
        risk_free_rate: Annualized risk-free rate for Sharpe ratio (default 0.0).

    Returns:
        JSON-serializable dict with either:
            {"status": "success", "metrics": {...}, "warnings": "..."}
            {"status": "error", "message": "..."}
    """
    # Step 1: Load CSV (impure — file I/O at the boundary)
    raw_df = load_csv(file_path)
    if raw_df is None:
        return {
            "status": "error",
            "message": f"Could not read CSV file: {file_path}",
        }

    # Step 2: Check that user's asset names exist in the CSV
    csv_columns = tuple(raw_df.columns)
    missing_assets = tuple(
        asset for asset in asset_names
        if asset not in csv_columns
    )
    if len(missing_assets) > 0:
        return {
            "status": "error",
            "message": (
                f"Assets not found in CSV: {', '.join(missing_assets)}. "
                f"Available columns: {', '.join(csv_columns)}."
            ),
        }

    # Step 3: Validate and clean data (pure)
    data_result: DataValidationResult = validate_data(raw_df)
    if not data_result.is_valid:
        return {
            "status": "error",
            "message": f"Data validation failed: {data_result.message}",
        }

    # Step 4: Validate weights against surviving assets (pure)
    weight_result: WeightValidationResult = validate_weights(
        weights=weights,
        original_assets=asset_names,
        surviving_assets=data_result.data.asset_names,
        risk_free_rate=risk_free_rate,
    )
    if not weight_result.is_valid:
        return {
            "status": "error",
            "message": f"Weight validation failed: {weight_result.message}",
        }

    # Step 5: Compute all metrics (pure)
    metrics: RiskMetrics = compute_all_metrics(data_result.data, weight_result.config)

    # Step 6: Convert to JSON-serializable dict (pure)
    # Collect any warnings from validation steps
    warnings = []
    if "dropped" in data_result.message.lower() or "filled" in data_result.message.lower():
        warnings.append(data_result.message)
    if "renormalize" in weight_result.message.lower() or "adjusted" in weight_result.message.lower():
        warnings.append(weight_result.message)

    return {
        "status": "success",
        "config": {
            "asset_names": list(weight_result.config.asset_names),
            "weights": list(weight_result.config.weights),
            "risk_free_rate": weight_result.config.risk_free_rate,
        },
        "metrics": metrics.to_dict(),
        "warnings": " ".join(warnings) if warnings else None,
    }
