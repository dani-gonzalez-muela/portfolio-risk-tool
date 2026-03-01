"""
Immutable data structures for the portfolio risk analysis pipeline.

All dataclasses are frozen (immutable once created) to enforce the functional
programming principle that data should not be modified after creation.
Instead, new instances are created at each pipeline stage:

    PortfolioConfig (user input)
        → DataValidationResult (validated data)
        → WeightValidationResult (validated weights)
        → RiskMetrics (output)
"""
from __future__ import annotations
from dataclasses import dataclass
import polars as pl


# ============================================================
# Input Structures
# ============================================================

@dataclass(frozen=True)
class PortfolioConfig:
    """
    User-defined portfolio configuration. Created from CLI arguments.

    Attributes:
        asset_names: Names of assets in the portfolio (e.g., ("ASSET_01", "ASSET_02"))
        weights: Allocation per asset, must sum to 1.0 (e.g., (0.5, 0.5))
        risk_free_rate: Annualized risk-free rate for Sharpe ratio calculation.
                        Defaults to 0.0 (excess returns = raw returns).
    """
    asset_names: tuple[str, ...]
    weights: tuple[float, ...]
    risk_free_rate: float = 0.0


# ============================================================
# Pipeline Data Container
# ============================================================

@dataclass(frozen=True)
class ReturnsData:
    """
    Validated and cleaned returns data. Created after loading and validating the CSV.

    This container signals that the data has passed all validation checks
    (no NaNs, correct columns, numeric values) and is safe for computation.

    Attributes:
        data: Polars DataFrame containing daily returns.
              # Polars note: unlike pandas, this DataFrame is immutable by default —
              # operations like .with_columns() return a NEW DataFrame
              # rather than modifying in place.
        asset_names: Column names matching the assets in the DataFrame.
        n_days: Number of trading days (rows) in the dataset.
        n_assets: Number of assets (columns) in the dataset.
    """
    data: pl.DataFrame
    asset_names: tuple[str, ...]
    n_days: int
    n_assets: int


# ============================================================
# Validation Result Types
# ============================================================
# FP approach to error handling: instead of raising exceptions (side effects),
# validators return explicit result objects that the caller can inspect.
# This keeps validation functions pure and testable.

@dataclass(frozen=True)
class DataValidationResult:
    """
    Result of data validation. Returned by validate_data().

    Attributes:
        is_valid: True if data passed validation (possibly with warnings).
        data: Validated ReturnsData if successful, None if validation failed.
        message: Describes what happened — errors, warnings about dropped
                 assets, or count of NaN values filled.
    """
    is_valid: bool
    data: ReturnsData | None
    message: str


@dataclass(frozen=True)
class WeightValidationResult:
    """
    Result of weight validation. Returned by validate_weights().

    If assets were dropped during data validation, the corresponding weights
    are removed and remaining weights are renormalized to sum to 1.0.

    Attributes:
        is_valid: True if weights are valid (possibly after renormalization).
        config: Validated PortfolioConfig if successful, None if validation failed.
        message: Describes what happened — errors or renormalization warnings.
    """
    is_valid: bool
    config: PortfolioConfig | None
    message: str


# ============================================================
# Output Structure
# ============================================================

@dataclass(frozen=True)
class RiskMetrics:
    """
    Computed risk metrics for the portfolio. Final output of the pipeline.

    All values are annualized where applicable (assuming 252 trading days/year).

    Attributes:
        portfolio_variance: Annualized portfolio variance (weighted combination of assets).
        annualized_return: Annualized portfolio return based on mean daily returns.
        sharpe_ratio: Risk-adjusted return = (annualized_return - risk_free_rate) / volatility.
        max_drawdown: Largest peak-to-trough decline in cumulative portfolio returns.
                      Expressed as a negative number (e.g., -0.15 means 15% drawdown).
        asset_volatilities: Per-asset annualized volatilities, one per asset.
        correlation_matrix: Pairwise correlation between assets.
                           # Stored as tuple of tuples rather than a numpy array or list
                           # to maintain immutability in the output container.
    """
    portfolio_variance: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    asset_volatilities: tuple[float, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict:
        """
        Convert metrics to a JSON-serializable dictionary.

        Pure transformation — no side effects, just restructures data.
        This is the final step before JSON output in the pipeline.
        """
        return {
            "portfolio_variance": self.portfolio_variance,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            # tuple → list because JSON only has arrays (no tuple concept).
            # This conversion happens ONLY here at the output boundary.
            # Inside the pipeline, everything stays as immutable tuples.
            "asset_volatilities": list(self.asset_volatilities),
            # Same idea: tuple of tuples → list of lists for JSON compatibility.
            "correlation_matrix": [list(row) for row in self.correlation_matrix],
        }
