"""
Test suite for data and weight validators.

Written BEFORE implementation (TDD: Red phase).
Validators use the FP approach: return result objects instead of raising exceptions.

Data validation rules:
    - Reject empty DataFrames
    - Reject non-numeric columns
    - Drop assets with >5% NaN values (warn user)
    - Fill remaining NaNs with 0.0 (assume no price movement)

Weight validation rules:
    - Number of weights must match number of (surviving) assets
    - Weights must sum to 1.0 (with floating point tolerance)
    - If assets were dropped, remove corresponding weights and renormalize
"""

import pytest
import polars as pl

# These imports WILL FAIL until we create validators.py — Red phase.
from portfolio_risk.validators import validate_data, validate_weights


# ============================================================
# Shared test data
# ============================================================

@pytest.fixture
def clean_data() -> pl.DataFrame:
    """Simple clean dataset with no issues."""
    return pl.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "ASSET_01": [0.01, 0.03, -0.01, 0.02],
        "ASSET_02": [0.02, 0.04, 0.01, -0.01],
    })


@pytest.fixture
def data_with_few_nans() -> pl.DataFrame:
    """Dataset with a few scattered NaNs (below 5% threshold per asset)."""
    return pl.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04",
                 "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08",
                 "2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12",
                 "2023-01-13", "2023-01-14", "2023-01-15", "2023-01-16",
                 "2023-01-17", "2023-01-18", "2023-01-19", "2023-01-20",
                 "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
                 "2023-01-25", "2023-01-26", "2023-01-27", "2023-01-28",
                 "2023-01-29", "2023-01-30"],
        # 1 NaN out of 30 rows = 3.3% — below 5% threshold, should be filled with 0.0
        "ASSET_01": [0.01, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, 0.01,
                     None, 0.02, 0.01, -0.01, 0.02, 0.03, -0.01, 0.01,
                     0.02, -0.01, 0.03, 0.01, 0.02, -0.01, 0.01, 0.03,
                     -0.02, 0.01, 0.02, -0.01, 0.03, 0.01],
        "ASSET_02": [0.02, 0.04, 0.01, -0.01, 0.02, 0.01, -0.03, 0.02,
                     0.01, -0.01, 0.02, 0.03, -0.01, 0.01, 0.02, -0.02,
                     0.01, 0.03, -0.01, 0.02, 0.01, -0.01, 0.02, 0.01,
                     0.03, -0.01, 0.02, 0.01, -0.02, 0.03],
    })


@pytest.fixture
def data_with_bad_asset() -> pl.DataFrame:
    """Dataset where ASSET_02 has >5% NaN values — should be dropped."""
    return pl.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04",
                 "2023-01-05", "2023-01-06", "2023-01-07", "2023-01-08",
                 "2023-01-09", "2023-01-10", "2023-01-11", "2023-01-12",
                 "2023-01-13", "2023-01-14", "2023-01-15", "2023-01-16",
                 "2023-01-17", "2023-01-18", "2023-01-19", "2023-01-20"],
        "ASSET_01": [0.01, 0.03, -0.01, 0.02, 0.01, -0.02, 0.03, 0.01,
                     0.02, -0.01, 0.01, 0.03, -0.01, 0.02, 0.01, -0.02,
                     0.03, 0.01, 0.02, -0.01],
        # 2 NaNs out of 20 rows = 10% — above 5% threshold, should be dropped
        "ASSET_02": [0.02, None, 0.01, -0.01, 0.02, None, -0.03, 0.02,
                     0.01, -0.01, 0.02, 0.03, -0.01, 0.01, 0.02, -0.02,
                     0.01, 0.03, -0.01, 0.02],
        "ASSET_03": [0.01, 0.02, -0.02, 0.01, 0.03, -0.01, 0.02, 0.01,
                     -0.01, 0.02, 0.01, -0.02, 0.03, 0.01, 0.02, -0.01,
                     0.01, 0.02, -0.01, 0.03],
    })


# ============================================================
# Data Validation Tests
# ============================================================

class TestValidateData:

    def test_clean_data_passes(self, clean_data):
        """Clean data with no issues should pass validation."""
        result = validate_data(clean_data)
        assert result.is_valid is True
        assert result.data is not None
        # date column should be excluded from asset columns
        assert "date" not in result.data.asset_names

    def test_empty_dataframe_fails(self):
        """Empty DataFrame should fail validation."""
        empty_df = pl.DataFrame()
        result = validate_data(empty_df)
        assert result.is_valid is False
        assert result.data is None

    def test_no_rows_fails(self):
        """DataFrame with columns but no rows should fail."""
        no_rows = pl.DataFrame({"ASSET_01": [], "ASSET_02": []}).cast(pl.Float64)
        result = validate_data(no_rows)
        assert result.is_valid is False
        assert result.data is None

    def test_non_numeric_column_excluded(self):
        """Non-numeric columns (other than 'date') should be excluded."""
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "ASSET_01": [0.01, 0.02, 0.03],
            "notes": ["good", "bad", "ok"],  # non-numeric, should be excluded
        })
        result = validate_data(df)
        assert result.is_valid is True
        assert "notes" not in result.data.asset_names
        assert "ASSET_01" in result.data.asset_names

    def test_few_nans_filled_with_zero(self, data_with_few_nans):
        """NaNs below 5% threshold should be filled with 0.0."""
        result = validate_data(data_with_few_nans)
        assert result.is_valid is True
        # After filling, there should be no NaN values in the data
        assert result.data.data.null_count().sum_horizontal().item() == 0
        # Message should mention how many NaNs were filled
        assert "filled" in result.message.lower() or "nan" in result.message.lower()

    def test_bad_asset_dropped(self, data_with_bad_asset):
        """Assets with >5% NaN should be dropped from the dataset."""
        result = validate_data(data_with_bad_asset)
        assert result.is_valid is True
        # ASSET_02 had 10% NaN — should be excluded
        assert "ASSET_02" not in result.data.asset_names
        # ASSET_01 and ASSET_03 should survive
        assert "ASSET_01" in result.data.asset_names
        assert "ASSET_03" in result.data.asset_names
        # Warning should mention the dropped asset
        assert "ASSET_02" in result.message

    def test_all_assets_dropped_fails(self):
        """If ALL assets exceed NaN threshold, validation should fail."""
        # 4 out of 20 rows = 20% NaN for both assets
        bad_data = pl.DataFrame({
            "date": [f"2023-01-{i:02d}" for i in range(1, 21)],
            "ASSET_01": [None, None, None, None] + [0.01] * 16,
            "ASSET_02": [None, None, None, None] + [0.02] * 16,
        })
        result = validate_data(bad_data)
        assert result.is_valid is False
        assert result.data is None

    def test_returns_correct_metadata(self, clean_data):
        """ReturnsData should have correct n_days, n_assets, and asset_names."""
        result = validate_data(clean_data)
        assert result.data.n_days == 4
        assert result.data.n_assets == 2
        assert result.data.asset_names == ("ASSET_01", "ASSET_02")


# ============================================================
# Weight Validation Tests
# ============================================================

class TestValidateWeights:

    def test_valid_weights(self):
        """Weights that sum to 1.0 with correct count should pass."""
        surviving_assets = ("ASSET_01", "ASSET_02")
        original_assets = ("ASSET_01", "ASSET_02")
        weights = (0.6, 0.4)
        result = validate_weights(weights, original_assets, surviving_assets)
        assert result.is_valid is True
        assert result.config is not None
        assert result.config.weights == (0.6, 0.4)

    def test_weights_sum_greater_than_one(self):
        """Weights summing to more than 1.0 should fail."""
        surviving_assets = ("ASSET_01", "ASSET_02")
        original_assets = ("ASSET_01", "ASSET_02")
        weights = (0.6, 0.6)  # sums to 1.2
        result = validate_weights(weights, original_assets, surviving_assets)
        assert result.is_valid is False
        assert result.config is None

    def test_weights_sum_less_than_one(self):
        """Weights summing to less than 1.0 should fail."""
        surviving_assets = ("ASSET_01", "ASSET_02")
        original_assets = ("ASSET_01", "ASSET_02")
        weights = (0.3, 0.2)  # sums to 0.5
        result = validate_weights(weights, original_assets, surviving_assets)
        assert result.is_valid is False
        assert result.config is None

    def test_wrong_number_of_weights(self):
        """Number of weights must match number of original assets."""
        surviving_assets = ("ASSET_01", "ASSET_02", "ASSET_03")
        original_assets = ("ASSET_01", "ASSET_02", "ASSET_03")
        weights = (0.5, 0.5)  # 2 weights for 3 assets
        result = validate_weights(weights, original_assets, surviving_assets)
        assert result.is_valid is False
        assert result.config is None

    def test_renormalize_after_asset_drop(self):
        """
        If an asset was dropped during data validation, its weight should be
        removed and remaining weights renormalized to sum to 1.0.
        """
        # User provided weights for 3 assets
        original_assets = ("ASSET_01", "ASSET_02", "ASSET_03")
        weights = (0.4, 0.2, 0.4)  # sums to 1.0

        # ASSET_02 was dropped during data validation
        surviving_assets = ("ASSET_01", "ASSET_03")

        result = validate_weights(weights, original_assets, surviving_assets)
        assert result.is_valid is True
        assert result.config is not None

        # Original weights for ASSET_01 and ASSET_03: 0.4, 0.4
        # After renormalization: 0.4/0.8 = 0.5, 0.4/0.8 = 0.5
        assert result.config.weights[0] == pytest.approx(0.5)
        assert result.config.weights[1] == pytest.approx(0.5)
        assert result.config.asset_names == ("ASSET_01", "ASSET_03")

        # Message should mention renormalization
        assert "renormalize" in result.message.lower() or "adjusted" in result.message.lower()

    def test_renormalize_unequal_weights(self):
        """Renormalization with unequal weights should preserve proportions."""
        original_assets = ("ASSET_01", "ASSET_02", "ASSET_03")
        weights = (0.5, 0.3, 0.2)

        # ASSET_02 dropped
        surviving_assets = ("ASSET_01", "ASSET_03")

        result = validate_weights(weights, original_assets, surviving_assets)
        assert result.is_valid is True

        # Surviving weights: 0.5, 0.2 → sum = 0.7
        # Renormalized: 0.5/0.7 ≈ 0.7143, 0.2/0.7 ≈ 0.2857
        assert result.config.weights[0] == pytest.approx(0.5 / 0.7, rel=1e-4)
        assert result.config.weights[1] == pytest.approx(0.2 / 0.7, rel=1e-4)

        # Should sum to 1.0 after renormalization
        assert sum(result.config.weights) == pytest.approx(1.0)

    def test_floating_point_tolerance(self):
        """Weights that are very close to 1.0 due to floating point should pass."""
        surviving_assets = ("ASSET_01", "ASSET_02", "ASSET_03")
        original_assets = ("ASSET_01", "ASSET_02", "ASSET_03")
        # These sum to 0.9999999999999999 due to floating point
        weights = (1/3, 1/3, 1/3)
        result = validate_weights(weights, original_assets, surviving_assets)
        assert result.is_valid is True
