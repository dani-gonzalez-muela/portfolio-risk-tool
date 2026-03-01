"""
Integration tests for the full pipeline.

These test the complete flow: CSV → validate → compute → output.
Unlike unit tests (test_metrics.py, test_validators.py) which test functions
in isolation, these verify that all pieces compose correctly together.
"""

import pytest
import polars as pl

from portfolio_risk.pipeline import run_pipeline, load_csv


# ============================================================
# Helper: Create temporary CSV files for testing
# ============================================================

@pytest.fixture
def simple_csv(tmp_path) -> str:
    """
    Create a simple CSV file with 2 assets, 4 days, no issues.
    tmp_path is a pytest built-in fixture that provides a temporary directory.
    """
    csv_path = tmp_path / "simple.csv"
    df = pl.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "ASSET_01": [0.01, 0.03, -0.01, 0.02],
        "ASSET_02": [0.02, 0.04, 0.01, -0.01],
    })
    df.write_csv(str(csv_path))
    return str(csv_path)


@pytest.fixture
def csv_with_nans(tmp_path) -> str:
    """CSV with a few NaN values (below threshold)."""
    csv_path = tmp_path / "with_nans.csv"
    df = pl.DataFrame({
        "date": [f"2023-01-{i:02d}" for i in range(1, 31)],
        "ASSET_01": [0.01 if i != 5 else None for i in range(30)],
        "ASSET_02": [0.02 if i != 10 else None for i in range(30)],
    })
    df.write_csv(str(csv_path))
    return str(csv_path)


# ============================================================
# Pipeline Integration Tests
# ============================================================

class TestRunPipeline:

    def test_success_simple(self, simple_csv):
        """Full pipeline with clean data should return success with all metrics."""
        result = run_pipeline(
            file_path=simple_csv,
            weights=(0.6, 0.4),
            asset_names=("ASSET_01", "ASSET_02"),
        )
        assert result["status"] == "success"
        assert "metrics" in result
        # Check all expected metrics are present
        metrics = result["metrics"]
        assert "portfolio_variance" in metrics
        assert "annualized_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "asset_volatilities" in metrics
        assert "correlation_matrix" in metrics

    def test_success_with_nans(self, csv_with_nans):
        """Pipeline should handle NaN filling and report warnings."""
        result = run_pipeline(
            file_path=csv_with_nans,
            weights=(0.5, 0.5),
            asset_names=("ASSET_01", "ASSET_02"),
        )
        assert result["status"] == "success"
        # Should have a warning about filled NaN values
        assert result["warnings"] is not None
        assert "filled" in result["warnings"].lower() or "nan" in result["warnings"].lower()

    def test_config_in_output(self, simple_csv):
        """Output should include the validated config for transparency."""
        result = run_pipeline(
            file_path=simple_csv,
            weights=(0.6, 0.4),
            asset_names=("ASSET_01", "ASSET_02"),
            risk_free_rate=0.02,
        )
        assert result["config"]["asset_names"] == ["ASSET_01", "ASSET_02"]
        assert result["config"]["weights"] == [0.6, 0.4]
        assert result["config"]["risk_free_rate"] == 0.02

    def test_error_bad_file(self):
        """Non-existent file should return an error, not crash."""
        result = run_pipeline(
            file_path="/nonexistent/path.csv",
            weights=(0.5, 0.5),
            asset_names=("ASSET_01", "ASSET_02"),
        )
        assert result["status"] == "error"
        assert "could not read" in result["message"].lower()

    def test_error_bad_weights(self, simple_csv):
        """Weights that don't sum to 1.0 should return an error."""
        result = run_pipeline(
            file_path=simple_csv,
            weights=(0.6, 0.6),  # sums to 1.2
            asset_names=("ASSET_01", "ASSET_02"),
        )
        assert result["status"] == "error"
        assert "weight" in result["message"].lower()

    def test_error_wrong_weight_count(self, simple_csv):
        """Weights referencing assets not in the CSV should return an error."""
        result = run_pipeline(
            file_path=simple_csv,
            weights=(0.5, 0.3, 0.2),  # 3 weights but CSV only has ASSET_01, ASSET_02
            asset_names=("ASSET_01", "ASSET_02", "ASSET_03"),
        )
        assert result["status"] == "error"
        assert "ASSET_03" in result["message"]


class TestLoadCsv:

    def test_valid_file(self, simple_csv):
        """Loading a valid CSV should return a DataFrame."""
        df = load_csv(simple_csv)
        assert df is not None
        assert df.height == 4
        assert df.width == 3  # date + 2 assets

    def test_nonexistent_file(self):
        """Loading a non-existent file should return None, not raise."""
        result = load_csv("/nonexistent/file.csv")
        assert result is None
