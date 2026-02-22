import asyncio
from src.ui.dashboard_tui import DashboardTUI
from rich.console import Console

async def test_dashboard():
    dashboard = DashboardTUI()
    
    # Mock Data with new fields
    mock_data = {
        "status": "Çalışıyor",
        "cycle": 123,
        "bankroll": 10500.0,
        "active_bets": 5,
        "win_rate": 0.65,
        "sharpe": 1.2,
        "drawdown": -0.02,
        "risk_level": "medium",
        "last_commentary": "🦁 [Nietzsche]: Cesaret, kaderi yener. Bahis alındı.",
        "signals": [
            {
                "match_id": "Galatasaray vs Fenerbahce",
                "selection": "1",
                "odds": 2.15,
                "ev": 0.08,
                "confidence": 0.75,
                "kelly_stake": 350.0,
                "phylosopher_approved": True
            },
            {
                "match_id": "Besiktas vs Trabzonspor",
                "selection": "X",
                "odds": 3.40,
                "ev": -0.02,
                "confidence": 0.45,
                "kelly_stake": 0,
                "phylosopher_approved": False
            },
            {
                "match_id": "Man City vs Liverpool",
                "selection": "2",
                "odds": 2.80,
                "ev": 0.12,
                "confidence": 0.82,
                "kelly_stake": 500.0,
                "phylosopher_approved": True
            }
        ]
    }
    
    dashboard.update(**mock_data)
    
    print("Dashboard created. Rendering one frame...")
    # Manually trigger render components to check for errors without full loop
    try:
        dashboard._render_header()
        print("✅ Header OK")
        dashboard._render_signals()
        print("✅ Signals OK")
        dashboard._render_metrics()
        print("✅ Metrics OK")
        dashboard._render_risk_panel()
        print("✅ Risk Panel OK")
        print("Dashboard Verification Successful")
        return True
    except Exception as e:
        print(f"❌ Dashboard Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        asyncio.run(test_dashboard())
    except KeyboardInterrupt:
        pass
