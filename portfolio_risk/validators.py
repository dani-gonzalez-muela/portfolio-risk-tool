"""
Pure validation functions for data and weight inputs.

FP approach to error handling: instead of raising exceptions (which are side effects),
these functions return explicit result objects (DataValidationResult, WeightValidationResult)
that the caller can inspect. This keeps all validation logic pure and testable.

Validation pipeline order:
    1. validate_data() — checks and cleans the raw DataFrame
    2. validate_weights() — checks weights against surviving assets, renormalizes if needed
"""

import polars as pl

from portfolio_risk.models import (
    DataValidationResult,
    PortfolioConfig,
    ReturnsData,
    WeightValidationResult,
)


# ============================================================
# Constants
# ============================================================

# Assets with more than this fraction of NaN values are dropped.
# 5% means if an asset has NaN in more than 1 out of every 20 days, it's unreliable.
NAN_THRESHOLD: float = 0.05

# Floating point tolerance for weight sum check.
# 1/3 + 1/3 + 1/3 = 0.9999999999999999 in Python, so we need a small tolerance.
WEIGHT_SUM_TOLERANCE: float = 1e-6


# ============================================================
# Data Validation
# ============================================================

def validate_data(raw_df: pl.DataFrame) -> DataValidationResult:
    """
    Validate and clean a raw DataFrame of asset returns.

    Steps:
        1. Check DataFrame is not empty
        2. Exclude non-numeric columns (except 'date' which is expected)
        3. Drop assets (columns) with >5% NaN values — warn user
        4. Fill remaining NaNs with 0.0 (assume no price movement)
        5. Wrap in ReturnsData container

    Missing data assumption: NaN values below the threshold are filled with 0.0,
    which assumes no price movement on that day. This is the correct return for
    days when markets are closed, and a conservative assumption for data gaps
    (slightly understates volatility). Assets exceeding the NaN threshold are
    considered unreliable and excluded entirely.

    Args:
        raw_df: Raw Polars DataFrame, typically from pl.read_csv().

    Returns:
        DataValidationResult with is_valid, cleaned ReturnsData (or None), and message.
    """
    # --- Check 1: DataFrame is not empty ---
    if raw_df.width == 0 or raw_df.height == 0:
        return DataValidationResult(
            is_valid=False,
            data=None,
            message="DataFrame is empty (no rows or no columns).",
        )

    # --- Check 2: Keep only numeric columns ---
    # Polars note: .select(cs.numeric()) selects only numeric columns.
    # We use a manual approach for clarity: check each column's dtype.
    # 'date' column is expected and always excluded from asset data.
    numeric_columns = tuple(
        col for col in raw_df.columns
        if col != "date" and raw_df[col].dtype.is_numeric()
    )

    if len(numeric_columns) == 0:
        return DataValidationResult(
            is_valid=False,
            data=None,
            message="No numeric asset columns found in DataFrame.",
        )

    # Create a new DataFrame with only numeric asset columns.
    # Polars note: .select() returns a NEW DataFrame (immutable operation).
    # In pandas this would be df[numeric_columns], which also creates a view/copy.
    numeric_df = raw_df.select(numeric_columns)

    # --- Check 3: Identify and drop assets with too many NaNs ---
    # Count NaN fraction per column
    nan_fractions = {
        col: numeric_df[col].null_count() / numeric_df.height
        for col in numeric_columns
    }

    # Separate assets into "keep" (below threshold) and "drop" (above threshold)
    # Using tuple comprehensions to stay immutable
    assets_to_drop = tuple(
        col for col, frac in nan_fractions.items()
        if frac > NAN_THRESHOLD
    )
    assets_to_keep = tuple(
        col for col in numeric_columns
        if col not in assets_to_drop
    )

    # Build warning messages
    messages = []

    if len(assets_to_drop) > 0:
        drop_details = ", ".join(
            f"{col} ({nan_fractions[col]:.1%} NaN)"
            for col in assets_to_drop
        )
        messages.append(f"Dropped assets exceeding {NAN_THRESHOLD:.0%} NaN threshold: {drop_details}.")

    # Check if any assets survive
    if len(assets_to_keep) == 0:
        return DataValidationResult(
            is_valid=False,
            data=None,
            message="All assets exceed NaN threshold — no valid data remaining.",
        )

    # Keep only surviving assets
    clean_df = numeric_df.select(assets_to_keep)

    # --- Check 4: Fill remaining NaNs with 0.0 ---
    # Count remaining NaNs before filling
    # Polars note: .null_count() returns count of null/NaN per column.
    # .sum_horizontal() sums across columns into a single value.
    remaining_nans = clean_df.null_count().sum_horizontal().item()

    if remaining_nans > 0:
        # Polars note: .fill_null(0.0) returns a NEW DataFrame with NaNs replaced.
        # The original clean_df is not modified (immutability).
        # In pandas you'd use df.fillna(0.0), which can modify in place or return new.
        clean_df = clean_df.fill_null(0.0)
        messages.append(f"Filled {remaining_nans} NaN values with 0.0 (assumed no price movement).")

    # --- Wrap in ReturnsData container ---
    if len(messages) == 0:
        messages.append("Data validation passed with no issues.")

    returns_data = ReturnsData(
        data=clean_df,
        asset_names=assets_to_keep,
        n_days=clean_df.height,
        n_assets=clean_df.width,
    )

    return DataValidationResult(
        is_valid=True,
        data=returns_data,
        message=" ".join(messages),
    )


# ============================================================
# Weight Validation
# ============================================================

def validate_weights(
    weights: tuple[float, ...],
    original_assets: tuple[str, ...],
    surviving_assets: tuple[str, ...],
    risk_free_rate: float = 0.0,
) -> WeightValidationResult:
    """
    Validate portfolio weights against the surviving asset list.

    If assets were dropped during data validation, the corresponding weights
    are removed and remaining weights are renormalized to sum to 1.0.
    This preserves the user's intended relative allocation.

    Example:
        Original: ASSET_01=0.4, ASSET_02=0.2, ASSET_03=0.4
        ASSET_02 dropped → surviving weights: 0.4, 0.4 → renormalized: 0.5, 0.5

    Args:
        weights: User-provided weights (one per original asset).
        original_assets: All asset names the user provided weights for.
        surviving_assets: Assets that passed data validation.
        risk_free_rate: Annualized risk-free rate for Sharpe calculation.

    Returns:
        WeightValidationResult with is_valid, PortfolioConfig (or None), and message.
    """
    # --- Check 1: Number of weights matches original assets ---
    if len(weights) != len(original_assets):
        return WeightValidationResult(
            is_valid=False,
            config=None,
            message=(
                f"Number of weights ({len(weights)}) does not match "
                f"number of assets ({len(original_assets)})."
            ),
        )

    # --- Check 2: Handle dropped assets ---
    # Map each original asset to its weight, then keep only surviving ones.
    # Using a dict comprehension (immutable operation — creates new dict).
    asset_weight_map = dict(zip(original_assets, weights))

    surviving_weights = tuple(
        asset_weight_map[asset]
        for asset in surviving_assets
        if asset in asset_weight_map
    )

    messages = []
    needs_renormalization = len(surviving_assets) < len(original_assets)

    if needs_renormalization:
        # Renormalize: scale surviving weights so they sum to 1.0
        # This preserves the user's intended relative allocation.
        weight_sum = sum(surviving_weights)

        if weight_sum == 0.0:
            return WeightValidationResult(
                is_valid=False,
                config=None,
                message="Surviving asset weights sum to 0.0 — cannot renormalize.",
            )

        # Create new tuple with renormalized weights (immutable)
        renormalized_weights = tuple(w / weight_sum for w in surviving_weights)

        dropped = set(original_assets) - set(surviving_assets)
        messages.append(
            f"Assets dropped during validation: {', '.join(sorted(dropped))}. "
            f"Weights renormalized from {weight_sum:.4f} to 1.0. "
            f"Adjusted weights: {', '.join(f'{w:.4f}' for w in renormalized_weights)}."
        )

        final_weights = renormalized_weights
    else:
        final_weights = surviving_weights

    # --- Check 3: Weights must sum to 1.0 (with tolerance) ---
    weight_sum = sum(final_weights)
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        return WeightValidationResult(
            is_valid=False,
            config=None,
            message=(
                f"Weights must sum to 1.0, but got {weight_sum:.6f}. "
                f"Difference: {abs(weight_sum - 1.0):.6f}."
            ),
        )

    # --- Build validated config ---
    if len(messages) == 0:
        messages.append("Weight validation passed.")

    config = PortfolioConfig(
        asset_names=surviving_assets,
        weights=final_weights,
        risk_free_rate=risk_free_rate,
    )

    return WeightValidationResult(
        is_valid=True,
        config=config,
        message=" ".join(messages),
    )
