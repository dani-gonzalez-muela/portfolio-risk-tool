# Portfolio Risk Analysis Tool

A command-line tool that analyzes a portfolio of assets for risk metrics, built with **functional programming principles** and **test-driven development**.

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd portfolio-risk-tool
pip install polars numpy pytest pytest-cov

# Run the tool
python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1

# Run tests
python -m pytest tests/ -v

# Run tests with coverage
python -m pytest tests/ --cov=portfolio_risk --cov-report=term-missing
```

## Example Output

```json
{
  "status": "success",
  "config": {
    "asset_names": ["ASSET_01", "ASSET_02", "ASSET_03", "..."],
    "weights": [0.1, 0.1, 0.1, "..."],
    "risk_free_rate": 0.0
  },
  "metrics": {
    "portfolio_variance": 0.004487,
    "annualized_return": 0.0815,
    "sharpe_ratio": 1.2167,
    "max_drawdown": -0.0871,
    "asset_volatilities": [0.0782, 0.1128, "..."],
    "correlation_matrix": [["..."]]
  },
  "warnings": "Filled 15 NaN values with 0.0 (assumed no price movement)."
}
```

## CLI Usage

```bash
# Basic usage (equal-weighted portfolio)
python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1

# With custom risk-free rate
python -m portfolio_risk --csv sample_returns.csv --weights 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 --risk-free-rate 0.02

# Help
python -m portfolio_risk --help
```

The tool auto-detects asset columns from the CSV. Weights are assigned in column order. If the weight count doesn't match, you'll get a friendly error listing the detected assets.

## Metrics Computed

All metrics are annualized assuming 252 trading days per year.

- **Portfolio Variance**: `σ² = wᵀΣw × 252` — weighted combination of asset covariances
- **Annualized Return**: `mean(daily portfolio returns) × 252`
- **Sharpe Ratio**: `(annualized return - risk-free rate) / annualized volatility` — returns 0.0 when volatility is zero
- **Max Drawdown**: Largest peak-to-trough decline in cumulative portfolio returns, computed via `functools.reduce`
- **Per-Asset Volatilities**: `std(daily returns, ddof=1) × √252` for each asset
- **Correlation Matrix**: Pairwise Pearson correlation between all assets

## Architecture

```
portfolio_risk/
├── models.py        # Frozen dataclasses (PortfolioConfig, ReturnsData, RiskMetrics)
├── metrics.py       # Pure metric functions (variance, Sharpe, drawdown, etc.)
├── validators.py    # Pure validation (data quality, weight checks)
├── pipeline.py      # Composition: load → validate → compute → output
├── cli.py           # Imperative shell: argparse, JSON output
└── __main__.py      # Entry point for python -m portfolio_risk

tests/
├── test_metrics.py      # 21 tests — hand-calculated expected values
├── test_validators.py   # 15 tests — data quality and weight edge cases
├── test_pipeline.py     #  8 tests — end-to-end integration
└── test_cli.py          #  6 tests — argument parsing and output format
```

**50 tests | 95% coverage**

## Design Principles

### Functional Programming

- **Immutability**: All data structures are `@dataclass(frozen=True)` with `tuple` fields instead of `list`. Polars DataFrames are immutable by default.
- **Pure functions**: Every function in `metrics.py` and `validators.py` takes typed inputs and returns typed outputs with no side effects.
- **Composability**: `compute_sharpe_ratio` composes `compute_annualized_return` and `compute_portfolio_variance` rather than reimplementing their logic. The pipeline chains small pure functions.
- **Functional error handling**: Validators return explicit result objects (`DataValidationResult`, `WeightValidationResult`) instead of raising exceptions.
- **Functional core, imperative shell**: All I/O (file reading, CLI args, printing) lives at the edges in `cli.py`. The core logic never touches the filesystem or prints anything.

### Polars over pandas

Polars was chosen because its DataFrames are immutable by default — every operation returns a new DataFrame rather than modifying in place. This aligns with the FP approach at the data layer, not just the function layer.

### Test-Driven Development

Tests were written before implementation (visible in commit history). Each metric has hand-calculated expected values using simple data that can be verified on paper.

## Data Validation

The tool handles real-world data quality issues:

- **Assets with >5% missing data**: Dropped entirely (unreliable), with a warning listing which assets were excluded
- **Remaining scattered NaNs**: Filled with 0.0 (assumes no price movement — correct for market closures, conservative for data gaps)
- **Weight renormalization**: If assets are dropped for data quality, corresponding weights are removed and remaining weights renormalized to preserve relative allocation
- **Non-numeric columns**: Automatically excluded
- **Friendly errors**: Wrong weight count tells you exactly how many assets were detected and their names

## Sample Data

`sample_returns.csv` contains 504 trading days (~2 years) of daily returns for 10 fictional assets with:
- Varying annualized volatilities (8% to 35%)
- Correlated returns (not independent noise)
- 15 intentional NaN values for testing data validation

Generated by `generate_sample_data.py` (included for reproducibility).
