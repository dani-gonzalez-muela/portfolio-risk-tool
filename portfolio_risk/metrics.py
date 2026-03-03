"""
Pure functions for portfolio risk metric computation.

Every function in this module is pure:
- Takes typed inputs, returns typed outputs
- No side effects (no file I/O, no printing, no mutation)
- Same inputs always produce same outputs

All metrics are annualized assuming 252 trading days per year.

Composability: compute_sharpe_ratio composes compute_annualized_return
and compute_portfolio_variance rather than reimplementing their logic.
This follows the FP principle of building complex functions from simple ones.
"""

from __future__ import annotations

import math
from functools import reduce

import polars as pl
import numpy as np


# ============================================================
# Constants
# ============================================================

TRADING_DAYS_PER_YEAR: int = 252


# ============================================================
# Asset-Level Metrics
# ============================================================

def compute_asset_volatilities(returns: pl.DataFrame) -> tuple[float, ...]:
    """
    Compute annualized volatility for each asset independently.

    Formula per asset:
        σ_annual = std(daily_returns, ddof=1) × √252

    Uses sample standard deviation (ddof=1) because we're estimating
    population volatility from a sample of observed returns.

    Args:
        returns: Polars DataFrame where each column is an asset's daily returns.

    Returns:
        Tuple of annualized volatilities, one per asset (immutable).
    """
    # Polars note: .std() defaults to ddof=1 (sample std), same as pandas.
    daily_stds = returns.select(
        # pl.all() selects every column — like applying a function across all assets.
        pl.all().std()
    )

    # Convert each daily std to annualized vol: multiply by √252
    # .row(0) extracts the first (only) row as a plain Python tuple
    annualization_factor = math.sqrt(TRADING_DAYS_PER_YEAR)
    daily_std_values = daily_stds.row(0)

    return tuple(std * annualization_factor for std in daily_std_values)


def compute_correlation_matrix(returns: pl.DataFrame) -> tuple[tuple[float, ...], ...]:
    """
    Compute pairwise Pearson correlation matrix between all assets.

    Args:
        returns: Polars DataFrame where each column is an asset's daily returns.

    Returns:
        Tuple of tuples (immutable 2D matrix). 
    """
    # Polars note: Polars' correlation API varies across versions, so we use
    # numpy's corrcoef which is stable and well-tested. We convert via .to_numpy()
    returns_matrix = returns.to_numpy()

    # np.corrcoef expects each ROW to be a variable, so we transpose.
    corr_array = np.corrcoef(returns_matrix.T)

    # Handle single asset: np.corrcoef returns a 0-d or weird-shaped array
    if corr_array.ndim == 0:
        corr_array = np.array([[1.0]])

    # Convert numpy array to tuple of tuples for immutability.
    return tuple(
        tuple(float(val) for val in row)
        for row in corr_array
    )


# ============================================================
# Portfolio-Level Metrics
# ============================================================

def compute_portfolio_variance(
    returns: pl.DataFrame,
    weights: tuple[float, ...],
) -> float:
    """
    Compute annualized portfolio variance using the covariance matrix approach.

    Formula:
        σ²_portfolio = wᵀ Σ w × 252
    where:
        w = weight vector
        Σ = sample covariance matrix of daily returns

    Args:
        returns: Polars DataFrame where each column is an asset's daily returns.
        weights: Tuple of asset weights (must match number of columns).

    Returns:
        Annualized portfolio variance (float).
    """
    # Convert to numpy for matrix math — Polars doesn't have built-in
    returns_matrix = returns.to_numpy()

    # np.cov expects each ROW to be a variable, so we transpose.
    cov_matrix = np.cov(returns_matrix.T, ddof=1)

    # Handle single asset: np.cov returns a scalar instead of a matrix
    if returns.width == 1:
        cov_matrix = np.array([[cov_matrix.item()]])

    weights_array = np.array(weights)

    # wᵀ Σ w — the classic portfolio variance formula
    daily_variance = float(weights_array @ cov_matrix @ weights_array)

    return daily_variance * TRADING_DAYS_PER_YEAR


def compute_annualized_return(
    returns: pl.DataFrame,
    weights: tuple[float, ...],
) -> float:
    """
    Compute annualized portfolio return from mean daily weighted returns.

    Formula:
        r_annual = mean(Σ wᵢ × rᵢ) × 252
    where wᵢ = weight of asset i, rᵢ = daily return of asset i.

    Args:
        returns: Polars DataFrame where each column is an asset's daily returns.
        weights: Tuple of asset weights.

    Returns:
        Annualized portfolio return (float).
    """
    # Compute weighted portfolio return for each day using Polars expressions.
    # pl.col(name) * weight creates a weighted column, then we sum across assets.
    asset_names = returns.columns
    weighted_sum_expr = sum(
        pl.col(name) * weight
        for name, weight in zip(asset_names, weights)
    )

    # .select() computes the expression and returns a new DataFrame (immutable).
    # .mean() aggregates to a single value.
    daily_mean = returns.select(
        weighted_sum_expr.alias("portfolio_return")
    ).mean().item()

    return daily_mean * TRADING_DAYS_PER_YEAR


def compute_sharpe_ratio(
    returns: pl.DataFrame,
    weights: tuple[float, ...],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute annualized Sharpe ratio (risk-adjusted return).

    Formula:
        Sharpe = (r_annual - r_f) / σ_annual

    Composes compute_annualized_return and compute_portfolio_variance
    rather than reimplementing their logic (FP composability principle).

    Convention: returns 0.0 when volatility is zero (avoids division by zero).

    Args:
        returns: Polars DataFrame where each column is an asset's daily returns.
        weights: Tuple of asset weights.
        risk_free_rate: Annualized risk-free rate (default 0.0).

    Returns:
        Annualized Sharpe ratio (float). Returns 0.0 if volatility is zero.
    """
    # Compose existing pure functions instead of recalculating
    annualized_ret = compute_annualized_return(returns, weights)
    portfolio_var = compute_portfolio_variance(returns, weights)
    annualized_vol = math.sqrt(portfolio_var)

    # Edge case: zero volatility (e.g., constant returns or all zeros)
    # Division by zero is undefined — return 0.0 by convention
    if annualized_vol == 0.0:
        return 0.0

    return (annualized_ret - risk_free_rate) / annualized_vol


def compute_max_drawdown(
    returns: pl.DataFrame,
    weights: tuple[float, ...],
) -> float:
    """
    Compute maximum drawdown: the largest peak-to-trough decline
    in cumulative portfolio returns.

    Uses functools.reduce to carry running state (peak, max_drawdown) through
    the sequence of returns — the functional alternative to a mutable loop variable.

    Formula at each step:
        cumulative_t = cumulative_{t-1} × (1 + r_t)
        peak_t = max(peak_{t-1}, cumulative_t)
        drawdown_t = (cumulative_t - peak_t) / peak_t

    Convention: expressed as a negative number (e.g., -0.15 = 15% drawdown).
    Returns 0.0 if there is never a decline.

    Args:
        returns: Polars DataFrame where each column is an asset's daily returns.
        weights: Tuple of asset weights.

    Returns:
        Maximum drawdown as a negative float (or 0.0 if no drawdown).
    """
    # First compute daily portfolio returns (weighted sum per day)
    asset_names = returns.columns
    weighted_sum_expr = sum(
        pl.col(name) * weight
        for name, weight in zip(asset_names, weights)
    )

    # .to_list() converts a Polars Series to a plain Python list.
    # We need this to feed into reduce, which works on iterables.
    daily_portfolio_returns = returns.select(
        weighted_sum_expr.alias("portfolio_return")
    ).to_series().to_list()

    # State carried through reduce: (cumulative_wealth, running_peak, max_drawdown)
    # Starting state: wealth = 1.0, peak = 1.0, no drawdown yet = 0.0
    initial_state = (1.0, 1.0, 0.0)

    def accumulate_drawdown(
        state: tuple[float, float, float],
        daily_return: float,
    ) -> tuple[float, float, float]:
        """
        Pure function that updates drawdown state with one new return.

        This is the function passed to reduce — it takes the previous state
        and one new data point, and returns the new state. No mutation.

        Args:
            state: (cumulative_wealth, running_peak, max_drawdown_so_far)
            daily_return: Single day's portfolio return.

        Returns:
            Updated state tuple (new cumulative, new peak, new max drawdown).
        """
        cumulative, peak, max_dd = state

        # Update cumulative wealth
        new_cumulative = cumulative * (1.0 + daily_return)

        # Update running peak (highest wealth seen so far)
        new_peak = max(peak, new_cumulative)

        # Current drawdown from peak (will be negative or zero)
        current_drawdown = (new_cumulative - new_peak) / new_peak

        # Track the worst (most negative) drawdown
        new_max_dd = min(max_dd, current_drawdown)

        return (new_cumulative, new_peak, new_max_dd)

    # reduce applies accumulate_drawdown across all daily returns,
    # carrying state from one day to the next. This is the FP alternative
    # to a for-loop with mutable variables.
    _, _, max_drawdown = reduce(accumulate_drawdown, daily_portfolio_returns, initial_state)

    return max_drawdown


def compute_sortino_ratio(
    returns: pl.DataFrame,
    weights: tuple[float, ...],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute annualized Sortino ratio (downside risk-adjusted return).

    Formula:
        Sortino = (r_annual - r_f) / downside_deviation

    Unlike Sharpe, which divides by total volatility, Sortino divides only by
    the volatility of NEGATIVE returns. This avoids penalizing upside volatility,
    making it a better measure for asymmetric return distributions.

    Composes compute_annualized_return (FP composability) and computes downside
    deviation from the negative returns only.

    Convention: returns 0.0 when downside deviation is zero (no negative returns).

    Args:
        returns: Polars DataFrame where each column is an asset's daily returns.
        weights: Tuple of asset weights.
        risk_free_rate: Annualized risk-free rate (default 0.0).

    Returns:
        Annualized Sortino ratio (float). Returns 0.0 if no negative returns.
    """
    # Compose: reuse annualized return calculation
    annualized_ret = compute_annualized_return(returns, weights)

    # Compute daily portfolio returns for downside deviation
    asset_names = returns.columns
    weighted_sum_expr = sum(
        pl.col(name) * weight
        for name, weight in zip(asset_names, weights)
    )

    daily_portfolio_returns = returns.select(
        weighted_sum_expr.alias("portfolio_return")
    ).to_series().to_list()

    # Filter to only negative returns using a comprehension (FP: no mutation)
    negative_returns = tuple(r for r in daily_portfolio_returns if r < 0)

    # Edge case: no negative returns → downside deviation is zero
    if len(negative_returns) < 2:
        return 0.0

    # Downside deviation: std of negative returns only × √252
    # Using ddof=1 (sample std) consistent with the rest of our metrics
    neg_array = np.array(negative_returns)
    downside_std = float(np.std(neg_array, ddof=1))
    downside_deviation = downside_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    if downside_deviation == 0.0:
        return 0.0

    return (annualized_ret - risk_free_rate) / downside_deviation


def compute_win_rate(
    returns: pl.DataFrame,
    weights: tuple[float, ...],
) -> float:
    """
    Compute the win rate: fraction of days with positive portfolio returns.

    Formula:
        win_rate = count(daily_return > 0) / total_days

    A simple but practical metric tracked by trading desks daily. Zero returns
    are NOT counted as wins (a flat day is not a winning day).

    Uses Polars filtering (FP: no loop, no mutation) to count positive days.

    Args:
        returns: Polars DataFrame where each column is an asset's daily returns.
        weights: Tuple of asset weights.

    Returns:
        Win rate as a float between 0.0 and 1.0.
    """
    asset_names = returns.columns
    weighted_sum_expr = sum(
        pl.col(name) * weight
        for name, weight in zip(asset_names, weights)
    )

    # Compute daily portfolio returns as a Polars Series
    # Polars note: .select() + .to_series() extracts a single column as a Series
    portfolio_returns = returns.select(
        weighted_sum_expr.alias("portfolio_return")
    ).to_series()

    # Count positive days using Polars expression (no loop)
    # Polars note: .filter() returns a NEW Series with only matching values
    n_positive = portfolio_returns.filter(portfolio_returns > 0).len()
    n_total = portfolio_returns.len()

    return n_positive / n_total
