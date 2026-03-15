import unittest
from unittest.mock import MagicMock, patch
import json
import pytest

from fastapi.testclient import TestClient

from src.ui.webapp_server import create_app, FASTAPI_OK

@pytest.mark.skipif(not FASTAPI_OK, reason="FastAPI is not installed")
class TestWebAppServer(unittest.TestCase):
    def setUp(self):
        # Create mock dependencies
        self.mock_db = MagicMock()
        self.mock_portfolio = MagicMock()
        self.mock_kalman = MagicMock()

        # Create app with mocks
        self.app = create_app(
            db=self.mock_db,
            portfolio=self.mock_portfolio,
            kalman=self.mock_kalman
        )
        self.client = TestClient(self.app)

        # Create app without mocks for default behavior
        self.app_no_mocks = create_app()
        self.client_no_mocks = TestClient(self.app_no_mocks)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "version": "1.0.0"})

    def test_portfolio_with_mock(self):
        self.mock_portfolio.status.return_value = {
            "bankroll": 5000,
            "total_pnl": 100,
            "win_rate": 0.5,
            "active_bets": [],
            "drawdown": 0,
            "sharpe": 1.5,
        }
        response = self.client.get("/api/portfolio")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["bankroll"], 5000)
        self.assertEqual(response.json()["total_pnl"], 100)
        self.mock_portfolio.status.assert_called_once()

    def test_portfolio_without_mock(self):
        response = self.client_no_mocks.get("/api/portfolio")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["bankroll"], 10000)

    def test_power_rankings_with_mock(self):
        self.mock_kalman.power_rankings.return_value = [
            {"rank": 1, "team": "A", "strength": 1500, "trend": "rising"}
        ]
        response = self.client.get("/api/power-rankings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 1)
        self.assertEqual(response.json()[0]["team"], "A")
        self.mock_kalman.power_rankings.assert_called_once_with(top_n=20)

    def test_power_rankings_without_mock(self):
        response = self.client_no_mocks.get("/api/power-rankings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_active_bets_with_mock(self):
        self.mock_db.get_active_bets.return_value = [{"id": 1, "match": "A vs B"}]
        response = self.client.get("/api/active-bets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["match"], "A vs B")
        self.mock_db.get_active_bets.assert_called_once()

    def test_active_bets_without_mock(self):
        response = self.client_no_mocks.get("/api/active-bets")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_signals_with_mock(self):
        self.mock_db.get_recent_signals.return_value = [{"id": 1, "signal": "BUY"}]
        response = self.client.get("/api/signals")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["signal"], "BUY")
        self.mock_db.get_recent_signals.assert_called_once_with(limit=20)

    def test_signals_without_mock(self):
        response = self.client_no_mocks.get("/api/signals")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_pnl_history_with_mock(self):
        self.mock_db.get_pnl_history.return_value = [{"date": "2026-02-01", "pnl": 50, "bankroll": 10050}]
        response = self.client.get("/api/pnl-history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["pnl"], 50)
        self.mock_db.get_pnl_history.assert_called_once_with(days=30)

    def test_pnl_history_without_mock(self):
        response = self.client_no_mocks.get("/api/pnl-history")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 16)
        self.assertEqual(response.json()[0]["bankroll"], 10000)

    @patch('src.ui.webapp_server.ROOT')
    def test_get_config_exists(self, mock_root):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = '{"kelly_fraction": 0.25}'
        mock_root.__truediv__.return_value = mock_path

        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"kelly_fraction": 0.25})

    @patch('src.ui.webapp_server.ROOT')
    def test_get_config_not_exists(self, mock_root):
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_root.__truediv__.return_value = mock_path

        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {})

    @patch('src.ui.webapp_server.ROOT')
    def test_post_config_exists(self, mock_root):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.return_value = '{"kelly_fraction": 0.25}'
        mock_root.__truediv__.return_value = mock_path

        response = self.client.post("/api/config", json={"kelly_fraction": 0.3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "updated": ["kelly_fraction"]})
        mock_path.write_text.assert_called_once()
        args, kwargs = mock_path.write_text.call_args
        written_data = json.loads(args[0])
        self.assertEqual(written_data["kelly_fraction"], 0.3)

    @patch('src.ui.webapp_server.ROOT')
    def test_post_config_not_exists(self, mock_root):
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_root.__truediv__.return_value = mock_path

        response = self.client.post("/api/config", json={"dd_limit": 15})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "updated": ["dd_limit"]})
        mock_path.write_text.assert_called_once()
        args, kwargs = mock_path.write_text.call_args
        written_data = json.loads(args[0])
        self.assertEqual(written_data["dd_limit"], 15)

    @patch('src.ui.webapp_server.ROOT')
    def test_post_config_exception(self, mock_root):
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.read_text.side_effect = Exception("Read Error")
        mock_root.__truediv__.return_value = mock_path

        response = self.client.post("/api/config", json={"kelly_fraction": 0.3})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "Read Error"})

    def test_index_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/html; charset=utf-8")
        self.assertIn("<title>Quant Bot \u2013 Dashboard</title>", response.text)

if __name__ == '__main__':
    unittest.main()
