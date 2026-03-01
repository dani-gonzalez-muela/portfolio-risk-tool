"""
Test suite for portfolio risk metrics.

These tests were written BEFORE the implementation (TDD: Red phase).
Each test uses simple, hand-calculable data so expected values can be
verified on paper. Tests will fail until metrics.py is implemented.

Convention: 252 trading days per year for annualization.
Convention: Zero volatility → Sharpe ratio returns 0.0 (avoids division by zero).
Convention: Max drawdown is expressed as a negative number (e.g., -0.15 = 15% drawdown).
"""

import pytest
import polars as pl

from portfolio_risk.metrics import (
    compute_portfolio_variance,
    compute_annualized_return,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_asset_volatilities,
    compute_correlation_matrix,
)


# ============================================================
# Shared test data (fixtures)
# ============================================================
# Using pytest fixtures so test data is defined once and reused.
# This avoids repetition and makes tests easier to maintain.

@pytest.fixture
def basic_returns() -> pl.DataFrame:
    """
    Simple 2-asset, 4-day returns dataset.
    Small enough to verify all calculations by hand.
    """
    return pl.DataFrame({
        "ASSET_01": [0.01, 0.03, -0.01, 0.02],
        "ASSET_02": [0.02, 0.04, 0.01, -0.01],
    })


@pytest.fixture
def basic_weights() -> tuple:
    """60/40 allocation across 2 assets."""
    return (0.6, 0.4)


@pytest.fixture
def single_asset_returns() -> pl.DataFrame:
    """Single asset — simplest possible case."""
    return pl.DataFrame({
        "ASSET_01": [0.01, 0.03, -0.01, 0.02],
    })


@pytest.fixture
def zero_returns() -> pl.DataFrame:
    """All returns are zero — tests edge cases around zero variance/volatility."""
    return pl.DataFrame({
        "ASSET_01": [0.0, 0.0, 0.0],
        "ASSET_02": [0.0, 0.0, 0.0],
    })


@pytest.fixture
def identical_assets_returns() -> pl.DataFrame:
    """Two assets with identical returns — correlation should be exactly 1.0."""
    return pl.DataFrame({
        "ASSET_01": [0.01, 0.03, -0.01, 0.02],
        "ASSET_02": [0.01, 0.03, -0.01, 0.02],
    })


# ============================================================
# Portfolio Variance Tests
# ============================================================
# Formula: σ²_portfolio = wᵀ Σ w  (annualized by × 252)
# where Σ = sample covariance matrix, w = weight vector

class TestPortfolioVariance:

    def test_basic_two_assets(self, basic_returns, basic_weights):
        """Basic case: 2 assets, 4 days, 60/40 weights."""
        # Hand-calculated: daily var = 0.00023033, annualized = 0.058044
        result = compute_portfolio_variance(basic_returns, basic_weights)
        assert result == pytest.approx(0.058044, rel=1e-4)

    def test_single_asset(self, single_asset_returns):
        """Single asset: portfolio variance = that asset's variance."""
        # Hand-calculated: sample var of [0.01, 0.03, -0.01, 0.02] × 252 = 0.0735
        result = compute_portfolio_variance(single_asset_returns, (1.0,))
        assert result == pytest.approx(0.0735, rel=1e-4)

    def test_zero_returns(self, zero_returns):
        """All zero returns: variance should be exactly 0."""
        result = compute_portfolio_variance(zero_returns, (0.5, 0.5))
        assert result == 0.0

    def test_equal_weights(self, basic_returns):
        """Equal weighting across assets."""
        # Just verifying it runs and returns a positive number
        result = compute_portfolio_variance(basic_returns, (0.5, 0.5))
        assert result > 0.0

    def test_negative_weights_short_position(self, basic_returns):
        """Negative weights (short positions) should still produce valid variance."""
        # Long ASSET_01, short ASSET_02. Reuses basic_returns fixture.
        weights = (1.2, -0.2)
        result = compute_portfolio_variance(basic_returns, weights)
        assert result > 0.0  # variance is always non-negative


# ============================================================
# Annualized Return Tests
# ============================================================
# Formula: annualized_return = mean(daily portfolio returns) × 252

class TestAnnualizedReturn:

    def test_basic_two_assets(self, basic_returns, basic_weights):
        """Basic case: weighted mean daily return × 252."""
        # Daily portfolio returns: [0.014, 0.034, -0.002, 0.008]
        # Mean = 0.0135, annualized = 0.0135 × 252 = 3.402
        result = compute_annualized_return(basic_returns, basic_weights)
        assert result == pytest.approx(3.402, rel=1e-4)

    def test_zero_returns(self, zero_returns):
        """All zero returns: annualized return should be 0."""
        result = compute_annualized_return(zero_returns, (0.5, 0.5))
        assert result == 0.0

    def test_single_asset(self, single_asset_returns):
        """Single asset: return = mean of that asset's returns × 252."""
        # Mean of [0.01, 0.03, -0.01, 0.02] = 0.0125, × 252 = 3.15
        result = compute_annualized_return(single_asset_returns, (1.0,))
        assert result == pytest.approx(3.15, rel=1e-4)


# ============================================================
# Sharpe Ratio Tests
# ============================================================
# Formula: sharpe = (annualized_return - risk_free_rate) / annualized_volatility
# Convention: if volatility = 0, return 0.0

class TestSharpeRatio:

    def test_basic_zero_risk_free(self, basic_returns, basic_weights):
        """Sharpe with risk-free rate = 0."""
        # annualized return = 3.402, vol = 0.24092, sharpe = 3.402 / 0.24092 = 14.1207
        result = compute_sharpe_ratio(basic_returns, basic_weights, risk_free_rate=0.0)
        assert result == pytest.approx(14.1207, rel=1e-3)

    def test_with_risk_free_rate(self, basic_returns, basic_weights):
        """Sharpe with non-zero risk-free rate."""
        # (3.402 - 0.02) / 0.24092 = 14.0377
        result = compute_sharpe_ratio(basic_returns, basic_weights, risk_free_rate=0.02)
        assert result == pytest.approx(14.0377, rel=1e-3)

    def test_zero_volatility_returns_zero(self):
        """When all returns are identical, volatility = 0 → Sharpe = 0.0 by convention."""
        # Constant returns: std = 0, so we can't divide
        constant_returns = pl.DataFrame({
            "ASSET_01": [0.01, 0.01, 0.01, 0.01],
        })
        result = compute_sharpe_ratio(constant_returns, (1.0,), risk_free_rate=0.0)
        assert result == 0.0

    def test_zero_returns(self, zero_returns):
        """Zero returns and zero vol: Sharpe should be 0.0."""
        result = compute_sharpe_ratio(zero_returns, (0.5, 0.5), risk_free_rate=0.0)
        assert result == 0.0

    def test_negative_average_return(self):
        """If portfolio loses money on average, Sharpe should be negative."""
        returns = pl.DataFrame({
            "ASSET_01": [-0.02, -0.03, 0.01, -0.04, -0.01],
        })
        result = compute_sharpe_ratio(returns, (1.0,), risk_free_rate=0.0)
        assert result < 0.0

    def test_returns_below_risk_free_rate(self):
        """Positive returns below risk-free rate should give negative Sharpe."""
        returns = pl.DataFrame({
            "ASSET_01": [0.0001, 0.0002, 0.0001, 0.0003, 0.0001],
        })
        # Very small positive returns, but risk-free rate is much higher
        result = compute_sharpe_ratio(returns, (1.0,), risk_free_rate=0.10)
        assert result < 0.0

    def test_negative_weights_short_position(self, basic_returns):
        """Sharpe ratio should work with short positions."""
        weights = (1.2, -0.2)
        result = compute_sharpe_ratio(basic_returns, weights, risk_free_rate=0.0)
        assert isinstance(result, float)


# ============================================================
# Max Drawdown Tests
# ============================================================
# Formula: track cumulative returns, find largest peak-to-trough decline.
# Convention: expressed as negative number (e.g., -0.05 = 5% drawdown).

class TestMaxDrawdown:

    def test_basic_two_assets(self, basic_returns, basic_weights):
        """Basic case with a small drawdown in the middle."""
        # Portfolio returns: [0.014, 0.034, -0.002, 0.008]
        # Cumulative: [1.014, 1.04848, 1.04638, 1.05475]
        # Drawdown occurs at day 3: (1.04638 - 1.04848) / 1.04848 = -0.002
        result = compute_max_drawdown(basic_returns, basic_weights)
        assert result == pytest.approx(-0.002, abs=1e-4)

    def test_always_positive_returns(self):
        """If returns are always positive, there's never a drawdown."""
        always_up = pl.DataFrame({
            "ASSET_01": [0.01, 0.02, 0.03, 0.01],
        })
        result = compute_max_drawdown(always_up, (1.0,))
        assert result == 0.0

    def test_always_negative_returns(self):
        """Continuous losses: drawdown keeps growing."""
        always_down = pl.DataFrame({
            "ASSET_01": [-0.05, -0.03, -0.04, -0.02],
        })
        # Initial wealth = 1.0, peak stays at 1.0 since all returns are negative.
        # Cumulative: [0.95, 0.9215, 0.88464, 0.86695]
        # Max drawdown at the end: (0.86695 - 1.0) / 1.0 = -0.13305
        result = compute_max_drawdown(always_down, (1.0,))
        assert result == pytest.approx(-0.13305, rel=1e-3)

    def test_zero_returns(self, zero_returns):
        """Zero returns: no movement, no drawdown."""
        result = compute_max_drawdown(zero_returns, (0.5, 0.5))
        assert result == 0.0

    def test_recovery_then_worse_drawdown(self):
        """
        Two drawdowns: first -15%, recovery to new peak, then -25%.
        Should find the worst one.

        Wealth trace:
            Day 1: +10%  → 1.100 (peak: 1.100)
            Day 2: -15%  → 0.935 (dd: -15.0%)
            Day 3: +20%  → 1.122 (new peak: 1.122)
            Day 4: -25%  → 0.842 (dd: -25.0% from peak — this is the worst)
            Day 5: +5%   → 0.884
        """
        returns = pl.DataFrame({
            "ASSET_01": [0.10, -0.15, 0.20, -0.25, 0.05],
        })
        result = compute_max_drawdown(returns, (1.0,))
        # Second drawdown is worse: (0.8415 - 1.122) / 1.122 = -0.25
        assert result == pytest.approx(-0.25, rel=1e-3)
        assert result < -0.20  # worse than the first -15% drawdown

    def test_total_loss(self):
        """A return of -100% means total loss. Drawdown should be -1.0."""
        returns = pl.DataFrame({
            "ASSET_01": [0.05, -1.0, 0.05],
        })
        result = compute_max_drawdown(returns, (1.0,))
        assert result == pytest.approx(-1.0)


# ============================================================
# Asset Volatilities Tests
# ============================================================
# Formula: per-asset annualized vol = std(daily returns, ddof=1) × √252

class TestAssetVolatilities:

    def test_basic_two_assets(self, basic_returns):
        """Check each asset's annualized volatility independently."""
        # ASSET_01: std([0.01, 0.03, -0.01, 0.02], ddof=1) × √252 = 0.27111
        # ASSET_02: std([0.02, 0.04, 0.01, -0.01], ddof=1) × √252 = 0.33045
        result = compute_asset_volatilities(basic_returns)
        assert result[0] == pytest.approx(0.27111, rel=1e-3)
        assert result[1] == pytest.approx(0.33045, rel=1e-3)

    def test_single_asset(self, single_asset_returns):
        """Single asset returns a tuple with one element."""
        result = compute_asset_volatilities(single_asset_returns)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.27111, rel=1e-3)

    def test_zero_returns(self, zero_returns):
        """Zero returns: volatility should be 0 for all assets."""
        result = compute_asset_volatilities(zero_returns)
        assert all(v == 0.0 for v in result)


# ============================================================
# Correlation Matrix Tests
# ============================================================
# Standard Pearson correlation between each pair of assets.

class TestCorrelationMatrix:

    def test_basic_two_assets(self, basic_returns):
        """Check shape and known correlation value."""
        result = compute_correlation_matrix(basic_returns)
        # Should be 2x2 matrix
        assert len(result) == 2
        assert len(result[0]) == 2
        # Diagonal should be 1.0 (asset correlated with itself)
        assert result[0][0] == pytest.approx(1.0)
        assert result[1][1] == pytest.approx(1.0)
        # Off-diagonal: hand-calculated = 0.32817
        assert result[0][1] == pytest.approx(0.32817, rel=1e-3)
        # Matrix should be symmetric
        assert result[0][1] == pytest.approx(result[1][0])

    def test_identical_assets(self, identical_assets_returns):
        """Identical return series: correlation should be exactly 1.0 everywhere."""
        result = compute_correlation_matrix(identical_assets_returns)
        assert result[0][1] == pytest.approx(1.0)
        assert result[1][0] == pytest.approx(1.0)

    def test_single_asset(self, single_asset_returns):
        """Single asset: 1x1 correlation matrix = [[1.0]]."""
        result = compute_correlation_matrix(single_asset_returns)
        assert len(result) == 1
        assert result[0][0] == pytest.approx(1.0)


# ============================================================
# Large Dataset Tests
# ============================================================
# Verifies metrics work at realistic scale, not just tiny test data.

class TestLargeDataset:

    def test_five_years_of_data(self):
        """1260 days (~5 years) should compute without errors."""
        import numpy as np
        rng = np.random.default_rng(42)
        n_days = 1260
        data = {
            "ASSET_01": rng.normal(0.0003, 0.01, n_days).tolist(),
            "ASSET_02": rng.normal(0.0002, 0.015, n_days).tolist(),
            "ASSET_03": rng.normal(0.0001, 0.02, n_days).tolist(),
        }
        returns = pl.DataFrame(data)
        weights = (0.5, 0.3, 0.2)

        variance = compute_portfolio_variance(returns, weights)
        sharpe = compute_sharpe_ratio(returns, weights)
        drawdown = compute_max_drawdown(returns, weights)

        assert variance > 0.0
        assert isinstance(sharpe, float)
        assert drawdown <= 0.0
