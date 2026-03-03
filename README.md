# Portfolio Risk Analysis Tool

A command-line tool that analyzes a portfolio of assets for risk metrics, built with **functional programming principles** and **test-driven development**.

## Quick Start

```bash
pip install polars numpy pytest pytest-cov hypothesis

# Run with CLI args
python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1

# Run with a config file
python -m portfolio_risk --config portfolio.json

# JSON output instead of summary
python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 --json

# Run tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=portfolio_risk --cov-report=term-missing
```

## Metrics

All metrics annualized assuming 252 trading days/year.

- **Portfolio Variance**: `wᵀΣw × 252` — weighted covariance
- **Annualized Return**: `mean(daily portfolio returns) × 252`
- **Sharpe Ratio**: `(return - risk_free) / volatility` — returns 0.0 when vol is zero
- **Sortino Ratio**: Like Sharpe, but only penalizes downside volatility
- **Max Drawdown**: Largest peak-to-trough decline, computed via `functools.reduce`
- **Win Rate**: Fraction of days with positive returns
- **Per-Asset Volatilities**: `std(daily, ddof=1) × √252`
- **Correlation Matrix**: Pairwise Pearson correlation between all assets

## Architecture

```
portfolio_risk/
├── models.py        # Frozen dataclasses (PortfolioConfig, ReturnsData, RiskMetrics)
├── metrics.py       # Pure metric functions (variance, Sharpe, Sortino, drawdown, etc.)
├── validators.py    # Pure validation (data quality checks, weight normalization)
├── pipeline.py      # Composition: load → validate → compute → output
├── cli.py           # Imperative shell: argparse, JSON/summary output
└── __main__.py      # Entry point for python -m portfolio_risk

tests/
├── test_metrics.py      # 42 tests — hand-calculated expected values
├── test_validators.py   # 18 tests — data quality and weight edge cases
├── test_pipeline.py     #  8 tests — end-to-end integration
├── test_cli.py          #  8 tests — argument parsing and output format
└── test_properties.py   # 17 tests — Hypothesis property-based testing
```

**89 tests total** including ~2000 randomized inputs from Hypothesis.

## Functional Programming Design

**Immutability**: All data structures use `@dataclass(frozen=True)` with `tuple` fields instead of `list`. Polars DataFrames are immutable by default — every operation returns a new DataFrame.

**Pure functions**: Every function in `metrics.py` and `validators.py` takes typed inputs and returns typed outputs with no side effects. No file I/O, no printing, no mutation.

**Composability**: The daily portfolio return series is computed once and shared across all downstream metrics (Sharpe, Sortino, drawdown, win rate). `compute_sharpe_ratio` composes `compute_annualized_return` and `compute_portfolio_variance` rather than reimplementing their logic. The pipeline chains small pure functions: `load → validate_data → validate_weights → compute_all_metrics → to_dict`.

**FP error handling**: Validators return explicit result objects (`DataValidationResult`, `WeightValidationResult`) instead of raising exceptions. This keeps validation functions pure and makes error paths testable.

**Functional core, imperative shell**: All I/O (file reading, CLI parsing, printing) lives at the edges in `cli.py` and `load_csv()`. The core logic never touches the filesystem.

**No loops**: `compute_max_drawdown` uses `functools.reduce` to carry state through the return sequence. Column operations use Polars expressions and comprehensions.

### Why Polars over pandas

Polars DataFrames are immutable by default — every operation returns a new DataFrame rather than modifying in place. This aligns with the FP approach at the data layer, not just the function layer.

## Testing Strategy

**Example-based tests** (`test_metrics.py`, `test_validators.py`): Hand-calculated expected values using simple data that can be verified on paper. Covers edge cases like zero returns, single assets, negative weights (short positions), total portfolio loss, floating point tolerance.

**Property-based tests** (`test_properties.py`): Uses Hypothesis to generate thousands of random portfolios and verify mathematical invariants — variance is always non-negative, drawdown is always in [-1, 0], correlations are always in [-1, 1], renormalized weights always sum to 1.0. During development, Hypothesis discovered that all-zero returns produce NaN correlations via `np.corrcoef` — an edge case that wouldn't surface from hand-written tests.

**Integration tests** (`test_pipeline.py`): End-to-end flow from CSV to output dict, verifying that all pieces compose correctly.

**CLI tests** (`test_cli.py`): Argument parsing, JSON output format, human-readable summary, config file mode, and error messages.

## Data Validation

- **Assets with >5% missing data**: Dropped entirely with a warning
- **Remaining NaNs**: Filled with 0.0 (assumes no price movement — correct for market closures, conservative for data gaps)
- **Weight renormalization**: If assets are dropped, corresponding weights are removed and remaining weights renormalized to preserve relative allocation
- **Non-numeric columns**: Silently excluded
- **Friendly errors**: Wrong weight count tells you exactly how many assets were detected and their names

## Sample Data

`sample_returns.csv` contains 504 trading days (~2 years) of daily returns for 10 fictional assets with varying volatilities (8%–35%), correlated returns, and 15 intentional NaN values. Generated by `generate_sample_data.py` for reproducibility.
