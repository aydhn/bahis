"""
webapp_server.py – Telegram Web App (Mini App) Sunucusu.

Telegram'da "Menü" butonuna basıldığında sohbet ekranının
yarısını kaplayan bir pencere açılır:
  - Canlı Kasa Grafiği (İnteraktif)
  - Risk Ayarı Değiştirme (Slider)
  - Aktif Kuponların Detaylı Listesi
  - Güç Sıralaması (Kalman)
  - Model Güven Aralıkları (Conformal)

Teknoloji: FastAPI + Jinja2 HTML + Vanilla JS
Telegram Web App API: window.Telegram.WebApp

Kurulum:
  1. BotFather'da /setmenubutton ile URL'yi ayarlayın
  2. HTTPS gerekli (ngrok veya cloudflare tunnel)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    FASTAPI_OK = True
except ImportError:
    FASTAPI_OK = False
    logger.info("fastapi/uvicorn yüklü değil – webapp devre dışı.")

ROOT = Path(__file__).resolve().parent.parent.parent


def create_app(db=None, portfolio=None, kalman=None,
               uncertainty=None, hedge_calc=None) -> Any:
    """FastAPI uygulaması oluştur."""
    if not FASTAPI_OK:
        logger.warning("[WebApp] fastapi yüklü değil.")
        return None

    app = FastAPI(
        title="Quant Betting Bot – TWA",
        description="Telegram Web App Dashboard",
        version="1.0.0",
    )

    # ═══════════════════════════════════════════
    #  API ENDPOINTS
    # ═══════════════════════════════════════════

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/api/portfolio")
    async def get_portfolio():
        """Portföy durumu – kasa, PnL, aktif bahisler."""
        if portfolio:
            status = portfolio.status()
        else:
            status = {
                "bankroll": 10000,
                "total_pnl": 0,
                "win_rate": 0,
                "active_bets": [],
                "drawdown": 0,
                "sharpe": 0,
            }
        return JSONResponse(status)

    @app.get("/api/power-rankings")
    async def get_power_rankings():
        """Kalman Filtresi güç sıralaması."""
        if kalman:
            rankings = kalman.power_rankings(top_n=20)
        else:
            rankings = []
        return JSONResponse(rankings)

    @app.get("/api/active-bets")
    async def get_active_bets():
        """Aktif kuponlar listesi."""
        if db and hasattr(db, "get_active_bets"):
            bets = db.get_active_bets()
        else:
            bets = []
        return JSONResponse(bets if isinstance(bets, list) else [])

    @app.get("/api/signals")
    async def get_signals():
        """Son sinyaller."""
        if db and hasattr(db, "get_recent_signals"):
            signals = db.get_recent_signals(limit=20)
        else:
            signals = []
        return JSONResponse(signals if isinstance(signals, list) else [])

    @app.get("/api/pnl-history")
    async def get_pnl_history():
        """PnL tarihçesi (grafik için)."""
        if db and hasattr(db, "get_pnl_history"):
            history = db.get_pnl_history(days=30)
        else:
            history = [
                {"date": f"2026-02-{i:02d}", "pnl": 0, "bankroll": 10000}
                for i in range(1, 17)
            ]
        return JSONResponse(history if isinstance(history, list) else [])

    @app.post("/api/config")
    async def update_config(request: Request):
        """Uzaktan konfigürasyon güncelleme."""
        body = await request.json()
        config_path = ROOT / "config.json"
        try:
            current = json.loads(config_path.read_text()) if config_path.exists() else {}
            current.update(body)
            config_path.write_text(json.dumps(current, indent=2))
            return {"status": "ok", "updated": list(body.keys())}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/config")
    async def get_config():
        """Mevcut konfigürasyon."""
        config_path = ROOT / "config.json"
        if config_path.exists():
            return JSONResponse(json.loads(config_path.read_text()))
        return JSONResponse({})

    # ═══════════════════════════════════════════
    #  ANA SAYFA – Telegram Web App HTML
    # ═══════════════════════════════════════════

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _render_webapp_html()

    return app


def _render_webapp_html() -> str:
    """Telegram Web App HTML/CSS/JS – tek dosya inline."""
    return """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quant Bot – Dashboard</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<style>
  :root {
    --bg: #0f1923;
    --card: #1a2733;
    --accent: #00d4aa;
    --accent2: #ff6b6b;
    --text: #e8eaed;
    --dim: #8b9dad;
    --border: #2a3a4a;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 12px;
  }
  .header {
    text-align: center;
    padding: 16px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
  }
  .header h1 { font-size: 18px; color: var(--accent); }
  .header .subtitle { font-size: 12px; color: var(--dim); margin-top: 4px; }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 16px;
  }
  .stat-card {
    background: var(--card);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    border: 1px solid var(--border);
  }
  .stat-card .value {
    font-size: 22px;
    font-weight: 700;
    color: var(--accent);
  }
  .stat-card .label {
    font-size: 11px;
    color: var(--dim);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .stat-card.negative .value { color: var(--accent2); }

  .section {
    background: var(--card);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    border: 1px solid var(--border);
  }
  .section h2 {
    font-size: 14px;
    color: var(--accent);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .chart-container {
    width: 100%;
    height: 160px;
    position: relative;
  }
  canvas { width: 100% !important; height: 100% !important; }

  .bet-list { list-style: none; }
  .bet-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
  }
  .bet-item:last-child { border-bottom: none; }
  .bet-match { font-size: 13px; font-weight: 600; }
  .bet-detail { font-size: 11px; color: var(--dim); }
  .bet-odds {
    background: var(--bg);
    padding: 4px 10px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 14px;
  }

  .slider-group {
    margin: 10px 0;
  }
  .slider-group label {
    font-size: 12px;
    color: var(--dim);
    display: flex;
    justify-content: space-between;
  }
  .slider-group input[type="range"] {
    width: 100%;
    margin-top: 6px;
    accent-color: var(--accent);
  }

  .ranking-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
  }
  .ranking-row:last-child { border-bottom: none; }
  .rank-num { color: var(--dim); width: 24px; }
  .rank-team { flex: 1; }
  .rank-strength { font-weight: 700; color: var(--accent); }
  .rank-trend { font-size: 11px; }
  .trend-up { color: #4caf50; }
  .trend-down { color: var(--accent2); }
  .trend-stable { color: var(--dim); }

  .refresh-btn {
    width: 100%;
    padding: 12px;
    background: var(--accent);
    color: var(--bg);
    border: none;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    margin-top: 8px;
  }
  .refresh-btn:active:not(:disabled) { opacity: 0.8; }
  .refresh-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .loading { text-align: center; color: var(--dim); padding: 20px; }
</style>
</head>
<body>

<div class="header">
  <h1>QUANT BETTING BOT</h1>
  <div class="subtitle">Level 9 – Reliability Dashboard</div>
</div>

<!-- Stats Grid -->
<div class="stats-grid">
  <div class="stat-card" id="card-bankroll">
    <div class="value" id="bankroll">--</div>
    <div class="label">Kasa (TL)</div>
  </div>
  <div class="stat-card" id="card-pnl">
    <div class="value" id="pnl">--</div>
    <div class="label">Gunluk PnL</div>
  </div>
  <div class="stat-card">
    <div class="value" id="win-rate">--</div>
    <div class="label">Win Rate</div>
  </div>
  <div class="stat-card">
    <div class="value" id="active-count">--</div>
    <div class="label">Aktif Kupon</div>
  </div>
</div>

<!-- PnL Chart -->
<div class="section">
  <h2>PnL Grafigi</h2>
  <div class="chart-container">
    <canvas id="pnlChart" role="img" aria-label="Kasa PnL Grafigi"></canvas>
  </div>
</div>

<!-- Risk Slider -->
<div class="section">
  <h2>Risk Ayari</h2>
  <div class="slider-group">
    <label for="kelly-slider">
      <span>Kelly Carpani</span>
      <span id="kelly-val">0.25</span>
    </label>
    <input type="range" id="kelly-slider" min="0.05" max="1.0" step="0.05" value="0.25">
  </div>
  <div class="slider-group">
    <label for="dd-slider">
      <span>Max Drawdown (%)</span>
      <span id="dd-val">10</span>
    </label>
    <input type="range" id="dd-slider" min="5" max="30" step="1" value="10">
  </div>
</div>

<!-- Active Bets -->
<div class="section">
  <h2>Aktif Kuponlar</h2>
  <ul class="bet-list" id="bet-list">
    <li class="loading">Yukleniyor...</li>
  </ul>
</div>

<!-- Power Rankings -->
<div class="section">
  <h2>Guc Siralamasi (Kalman)</h2>
  <div id="rankings">
    <div class="loading">Yukleniyor...</div>
  </div>
</div>

<button class="refresh-btn" onclick="refreshAll()" aria-label="Verileri yenile" aria-busy="false" aria-live="polite">Yenile</button>

<script>
const tg = window.Telegram?.WebApp;
if (tg) {
  tg.ready();
  tg.expand();
  tg.MainButton.hide();
}

const API = '';

async function fetchJSON(url) {
  try {
    const r = await fetch(API + url);
    return await r.json();
  } catch(e) { return null; }
}

async function loadPortfolio() {
  const d = await fetchJSON('/api/portfolio');
  if (!d) return;
  document.getElementById('bankroll').textContent =
    (d.bankroll || 10000).toLocaleString('tr-TR', {maximumFractionDigits: 0});
  const pnl = d.total_pnl || 0;
  const pnlEl = document.getElementById('pnl');
  pnlEl.textContent = (pnl >= 0 ? '+' : '') + pnl.toFixed(0);
  document.getElementById('card-pnl').className =
    'stat-card' + (pnl < 0 ? ' negative' : '');
  document.getElementById('win-rate').textContent =
    ((d.win_rate || 0) * 100).toFixed(0) + '%';
  const bets = d.active_bets || [];
  document.getElementById('active-count').textContent = bets.length || '0';
}

async function loadBets() {
  const bets = await fetchJSON('/api/active-bets');
  const list = document.getElementById('bet-list');
  if (!bets || bets.length === 0) {
    list.innerHTML = '<li class="bet-item"><span class="bet-detail">Aktif kupon yok</span></li>';
    return;
  }
  list.innerHTML = bets.map(b => `
    <li class="bet-item">
      <div>
        <div class="bet-match">${b.match || b.match_id || '?'}</div>
        <div class="bet-detail">${b.selection || ''} | EV: ${(b.ev || 0).toFixed(1)}%</div>
      </div>
      <div class="bet-odds">${(b.odds || 0).toFixed(2)}</div>
    </li>
  `).join('');
}

async function loadRankings() {
  const ranks = await fetchJSON('/api/power-rankings');
  const el = document.getElementById('rankings');
  if (!ranks || ranks.length === 0) {
    el.innerHTML = '<div class="ranking-row"><span class="bet-detail">Veri yok</span></div>';
    return;
  }
  el.innerHTML = ranks.slice(0, 10).map(r => {
    const trendClass = r.trend === 'rising' ? 'trend-up' :
                       r.trend === 'falling' ? 'trend-down' : 'trend-stable';
    const arrow = r.trend === 'rising' ? '&#9650;' :
                  r.trend === 'falling' ? '&#9660;' : '&#9679;';
    return `
      <div class="ranking-row">
        <span class="rank-num">${r.rank}.</span>
        <span class="rank-team">${r.team}</span>
        <span class="rank-strength">${r.strength}</span>
        <span class="rank-trend ${trendClass}">${arrow}</span>
      </div>
    `;
  }).join('');
}

async function loadPnLChart() {
  const data = await fetchJSON('/api/pnl-history');
  if (!data || data.length === 0) return;

  const canvas = document.getElementById('pnlChart');
  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.offsetWidth;
  const H = 160;
  canvas.width = W * 2;
  canvas.height = H * 2;
  ctx.scale(2, 2);

  const values = data.map(d => d.bankroll || d.pnl || 0);
  const min = Math.min(...values) * 0.95;
  const max = Math.max(...values) * 1.05;
  const range = max - min || 1;

  ctx.clearRect(0, 0, W, H);

  // Gradient fill
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(0, 212, 170, 0.3)');
  grad.addColorStop(1, 'rgba(0, 212, 170, 0.0)');

  ctx.beginPath();
  ctx.moveTo(0, H);
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * W;
    const y = H - ((v - min) / range) * (H - 20) - 10;
    ctx.lineTo(x, y);
  });
  ctx.lineTo(W, H);
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1)) * W;
    const y = H - ((v - min) / range) * (H - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#00d4aa';
  ctx.lineWidth = 2;
  ctx.stroke();
}

// Slider events
document.getElementById('kelly-slider').addEventListener('input', function() {
  document.getElementById('kelly-val').textContent = this.value;
});
document.getElementById('dd-slider').addEventListener('input', function() {
  document.getElementById('dd-val').textContent = this.value;
});

async function refreshAll() {
  const btn = document.querySelector('.refresh-btn');
  if (btn) {
    btn.disabled = true;
    btn.setAttribute('aria-busy', 'true');
    btn.textContent = 'Yenileniyor...';
  }

  try {
    await Promise.all([
      loadPortfolio(),
      loadBets(),
      loadRankings(),
      loadPnLChart()
    ]);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.setAttribute('aria-busy', 'false');
      btn.textContent = 'Yenile';
    }
  }
}

// Initial load
refreshAll();
setInterval(refreshAll, 30000);
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════
#  SUNUCU BAŞLATMA
# ═══════════════════════════════════════════════
def start_webapp(host: str = "0.0.0.0", port: int = 8080, **kwargs):
    """Web App sunucusunu başlat."""
    if not FASTAPI_OK:
        logger.error("[WebApp] fastapi/uvicorn yüklü değil.")
        return

    app = create_app(**kwargs)
    if app is None:
        return

    logger.info(f"[WebApp] Telegram Mini App başlatılıyor → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


async def start_webapp_async(host: str = "0.0.0.0", port: int = 8080, **kwargs):
    """Asyncio event loop içinden başlat."""
    if not FASTAPI_OK:
        return

    app = create_app(**kwargs)
    if app is None:
        return

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    logger.info(f"[WebApp] TWA sunucu başlatıldı → http://{host}:{port}")
    await server.serve()


if __name__ == "__main__":
    start_webapp()
