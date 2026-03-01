"""
Tests for the CLI module.

These test the imperative shell: argument parsing and JSON output.
We test by calling main() with mocked sys.argv rather than spawning subprocesses.
"""

import json
import sys

import pytest

from portfolio_risk.cli import build_parser, main


class TestBuildParser:

    def test_required_args(self):
        """Parser should require --csv and --weights."""
        parser = build_parser()
        # Missing required args should raise SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_basic_args(self):
        """Parser should correctly parse csv, weights, and risk-free-rate."""
        parser = build_parser()
        args = parser.parse_args([
            "--csv", "test.csv",
            "--weights", "0.6", "0.4",
            "--risk-free-rate", "0.02",
        ])
        assert args.csv == "test.csv"
        assert args.weights == [0.6, 0.4]
        assert args.risk_free_rate == 0.02

    def test_default_risk_free_rate(self):
        """Risk-free rate should default to 0.0."""
        parser = build_parser()
        args = parser.parse_args(["--csv", "test.csv", "--weights", "1.0"])
        assert args.risk_free_rate == 0.0


class TestMain:

    def test_bad_file_prints_error(self, capsys, monkeypatch):
        """main() should print JSON error for non-existent file."""
        # monkeypatch replaces sys.argv so argparse reads our test args.
        # capsys captures stdout so we can check the JSON output.
        monkeypatch.setattr(sys, "argv", [
            "portfolio_risk", "--csv", "/nonexistent.csv", "--weights", "0.5", "0.5",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "error"

    def test_success_output_is_valid_json(self, capsys, monkeypatch, tmp_path):
        """main() should print valid JSON on success."""
        import polars as pl

        # Create a temp CSV
        csv_path = tmp_path / "test.csv"
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            "ASSET_01": [0.01, 0.03, -0.01, 0.02],
            "ASSET_02": [0.02, 0.04, 0.01, -0.01],
        })
        df.write_csv(str(csv_path))

        monkeypatch.setattr(sys, "argv", [
            "portfolio_risk", "--csv", str(csv_path), "--weights", "0.6", "0.4",
        ])
        main()

        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "success"
        assert "metrics" in output

    def test_wrong_weight_count_prints_friendly_error(self, capsys, monkeypatch, tmp_path):
        """Wrong number of weights should show detected assets in error message."""
        import polars as pl

        csv_path = tmp_path / "test.csv"
        df = pl.DataFrame({
            "date": ["2023-01-01", "2023-01-02"],
            "ASSET_01": [0.01, 0.02],
            "ASSET_02": [0.03, 0.04],
            "ASSET_03": [0.05, 0.06],
        })
        df.write_csv(str(csv_path))

        monkeypatch.setattr(sys, "argv", [
            "portfolio_risk", "--csv", str(csv_path), "--weights", "0.5", "0.5",
        ])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        output = json.loads(capsys.readouterr().out)
        assert output["status"] == "error"
        # Should tell the user how many assets were found
        assert "3" in output["message"]
        assert "2" in output["message"]
