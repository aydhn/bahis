import sys
from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timezone
from pathlib import Path

from src.utils.strategy_health_report import StrategyHealthReport

@pytest.fixture
def mock_reports_dir(tmp_path):
    with patch("src.utils.strategy_health_report.REPORTS_DIR", tmp_path):
        yield tmp_path

def test_initialization(mock_reports_dir):
    """Test history initialization and directory creation."""
    report = StrategyHealthReport()
    assert report._history == []
    assert report._signals_history == []
    assert mock_reports_dir.exists()

def test_update(mock_reports_dir):
    """Test updating history with valid bets and skips."""
    report = StrategyHealthReport()
    bets = [
        {"match_id": "1", "selection": "home", "ev": 0.1, "confidence": 0.6, "stake_pct": 0.05},
        {"match_id": "2", "selection": "skip", "ev": -0.05, "confidence": 0.3, "stake_pct": 0.0},
        {"match_id": "3", "selection": "away", "ev": 0.2, "confidence": 0.8, "stake_pct": 0.1},
    ]
    ensemble = [] # unused in the method for now

    report.update(bets, ensemble)

    assert len(report._history) == 1
    assert len(report._signals_history) == 3

    record = report._history[0]
    assert "timestamp" in record
    assert record["n_bets"] == 2 # 2 non-skip bets
    # avg ev includes skips as per current logic
    assert abs(record["avg_ev"] - ((0.1 - 0.05 + 0.2) / 3)) < 1e-6
    assert abs(record["avg_confidence"] - ((0.6 + 0.3 + 0.8) / 3)) < 1e-6
    assert abs(record["total_stake"] - 0.15) < 1e-6

def test_compute_metrics_empty(mock_reports_dir):
    report = StrategyHealthReport()
    metrics = report._compute_metrics()
    assert metrics == {"status": "Henüz veri yok"}

def test_compute_metrics_with_data(mock_reports_dir):
    report = StrategyHealthReport()
    report._history = [
        {"n_bets": 2, "avg_ev": 0.1, "avg_confidence": 0.6, "total_stake": 0.1},
        {"n_bets": 3, "avg_ev": 0.2, "avg_confidence": 0.7, "total_stake": 0.15},
    ]
    metrics = report._compute_metrics()
    assert metrics["toplam_dongu"] == 2
    assert metrics["toplam_bahis"] == 5
    assert metrics["ortalama_ev"] == "0.1500"
    assert metrics["ortalama_guven"] == "65.00%"
    assert metrics["toplam_stake"] == "0.2500"
    assert float(metrics["sharpe_ratio"]) > 0

def test_generate_text_report(mock_reports_dir):
    report = StrategyHealthReport()
    report._history = [{"n_bets": 2, "avg_ev": 0.1, "avg_confidence": 0.6, "total_stake": 0.1}]

    path_str = report._generate_text_report()
    path = Path(path_str)

    assert path.exists()
    assert path.suffix == ".txt"
    content = path.read_text(encoding="utf-8")
    assert "QUANT BETTING BOT – STRATEJİ SAĞLIK RAPORU" in content
    assert "Toplam Dongu" in content

def test_generate_pdf_without_fpdf(mock_reports_dir):
    """Test generating PDF when fpdf is not installed."""
    report = StrategyHealthReport()
    report._history = [{"n_bets": 2, "avg_ev": 0.1, "avg_confidence": 0.6, "total_stake": 0.1}]

    # Mock sys.modules to simulate ImportError for fpdf
    with patch.dict(sys.modules, {'fpdf': None}):
        path_str = report.generate_pdf()
        path = Path(path_str)
        # Should fallback to text report
        assert path.exists()
        assert path.suffix == ".txt"

def test_generate_pdf_with_fpdf(mock_reports_dir):
    """Test generating PDF successfully when fpdf is installed."""
    report = StrategyHealthReport()
    report._history = [{"n_bets": 2, "avg_ev": 0.1, "avg_confidence": 0.6, "total_stake": 0.1}]
    report._signals_history = [
        {"match_id": "1", "selection": "home", "ev": 0.1, "confidence": 0.6, "stake_pct": 0.05}
    ]

    # Create a mock FPDF class and instance
    mock_fpdf_instance = MagicMock()
    mock_fpdf_class = MagicMock(return_value=mock_fpdf_instance)

    mock_fpdf_module = MagicMock()
    mock_fpdf_module.FPDF = mock_fpdf_class

    with patch.dict(sys.modules, {'fpdf': mock_fpdf_module}):
        path_str = report.generate_pdf()

        # Verify the FPDF methods were called correctly
        mock_fpdf_instance.add_page.assert_called_once()
        mock_fpdf_instance.set_font.assert_called()
        mock_fpdf_instance.cell.assert_called()
        mock_fpdf_instance.output.assert_called_once()

        # Verify output path
        path = Path(path_str)
        assert path.suffix == ".pdf"
        assert path.parent == mock_reports_dir

def test_console_summary(mock_reports_dir):
    """Test printing summary to console."""
    report = StrategyHealthReport()
    report._history = [{"n_bets": 2, "avg_ev": 0.1, "avg_confidence": 0.6, "total_stake": 0.1}]

    with patch("src.utils.strategy_health_report.Console") as mock_console_class:
        mock_console = mock_console_class.return_value
        report.console_summary()
        mock_console.print.assert_called_once()
        # The first argument should be a rich.table.Table instance
        args, _ = mock_console.print.call_args
        assert args[0].__class__.__name__ == "Table"
