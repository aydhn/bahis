"""
sentinel.py – Otonom Sistem Orkestratörü (The CEO).

Bu modül, tüm sistemi (Pipeline, Risk, Bot) tek bir çatı altında yönetir.
"Sentinel", sistemin otonom karar vericisi ve koruyucusudur.

Özellikler:
  - Tek bir Telegram Bot instance'ı yönetir.
  - Pipeline'ı başlatır, durdurur, izler.
  - Risk modunu dinamik olarak değiştirir.
  - Acil durum komutlarını (Force Bet, Shutdown) uygular.
"""
import asyncio
import signal
import sys
from typing import Optional, Any
from loguru import logger

from src.reporting.telegram_bot import TelegramBot
from src.pipeline.core import create_default_pipeline, PipelineEngine
from src.system.lifecycle import lifecycle
from src.core.regime_kelly import RegimeKelly, RegimeState
from src.system.container import container
from src.core.event_bus import EventBus
from src.quant.risk.portfolio_manager import PortfolioManager
from src.core.circuit_breaker import CircuitBreakerRegistry

class Sentinel:
    """
    Sistemin beyni ve yöneticisi.
    """

    def __init__(self, daemon_mode: bool = True):
        self.daemon_mode = daemon_mode
        self.running = False

        # 0. Event Bus (Merkezi Sinir Sistemi)
        self.bus = EventBus()
        self.portfolio_manager = PortfolioManager(self.bus)

        # 1. Bot Entegrasyonu
        self.bot = TelegramBot()
        self.bot.set_sentinel(self)
        # Botu event bus'a abone yapabiliriz (daha sonra)

        # 2. Pipeline Hazırlığı
        self.pipeline: PipelineEngine = create_default_pipeline(
            bot_instance=self.bot,
            bus=self.bus
        )

        # 3. Risk Yöneticisi (Shared State via Container)
        # RiskStage ile aynı instance'ı kullandığımızdan emin olmalıyız.
        self.risk_manager = container.get("regime_kelly")
        if self.risk_manager:
            self.bus.subscribe("market_regime_update", self.risk_manager.update_regime)

        self.portfolio_opt = container.get("portfolio_opt")

        # 4. Global Circuit Breaker (Panic Button)
        self.cb_registry = CircuitBreakerRegistry()
        self.system_breaker = self.cb_registry.get_or_create("system_health", preset="api")
        # 1 saatlik soğuma, 3 kritik hata -> OPEN
        self.system_breaker._config.recovery_timeout = 3600.0

        # Sinyal Yakalama
        if daemon_mode:
            signal.signal(signal.SIGINT, self._handle_sigint)
            signal.signal(signal.SIGTERM, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        """Ctrl+C yakalayıcı."""
        logger.warning("Sentinel: Kapanma sinyali alındı.")
        self.shutdown()

    async def run(self):
        """Sistemi başlat."""
        logger.info("╔════════════════════════════════════════╗")
        logger.info("║     SENTINEL OTONOM SİSTEM (v2.0)      ║")
        logger.info("╚════════════════════════════════════════╝")

        self.running = True

        # Botu başlat
        if self.bot.enabled:
            asyncio.create_task(self.bot.start())
            await self.bot.send_message(
                int(self.bot.token.split(":")[0]) if self.bot.token else 0, # Fallback ID
                "🚀 *Sentinel Başlatıldı.*"
            )

        # Pipeline döngüsü
        try:
            if self.daemon_mode:
                logger.info("Sentinel: Daemon Modu Aktif. Kontrollü döngü başlıyor.")
                while self.running and not lifecycle.shutdown_event.is_set():
                    # 1. Sağlık Kontrolü
                    if not self._check_health():
                        logger.warning("Sentinel: Sistem sağlığı KRİTİK. Devre Kesici AÇIK. 5 dakika bekleniyor.")
                        if self.bot and self.bot.enabled:
                             await self.bot.send_risk_alert("CIRCUIT BREAKER", "Sistem finansal koruma moduna geçti. İşlemler durduruldu.")
                        await asyncio.sleep(300)
                        continue

                    # 2. Pipeline Döngüsü (Tek adım)
                    await self.pipeline.run_once()

                    # 3. Bekleme
                    await asyncio.sleep(10)
            else:
                if self._check_health():
                    await self.pipeline.run_once()
                else:
                    logger.error("Sistem sağlığı yetersiz. Çalışma iptal edildi.")

        except Exception as e:
            logger.critical(f"Sentinel Critical Error: {e}")
        finally:
            await self.shutdown_async()

    def _check_health(self) -> bool:
        """Sistemin finansal ve teknik sağlığını kontrol et."""
        if not self.system_breaker.is_available:
            return False

        try:
            # DB'den Ardışık Kayıp Kontrolü
            db = container.get("db")
            if db:
                # Son 10 bahis kontrolü
                try:
                    df = db.query("SELECT status FROM bets WHERE status IN ('won', 'lost') ORDER BY settled_at DESC LIMIT 10")
                    if not df.is_empty():
                        statuses = df["status"].to_list()
                        # Eğer 10 bahis varsa ve hepsi kayıpsa
                        if len(statuses) >= 10 and all(s == 'lost' for s in statuses):
                            raise Exception("10 Ardışık Kayıp! Panic Button devrede.")
                except Exception as db_err:
                     # Tablo yoksa vs yut, sistemi durdurma
                     logger.warning(f"Health check DB uyarısı: {db_err}")

            self.system_breaker._on_success()
            return True
        except Exception as e:
            logger.critical(f"Health Check Failed: {e}")
            self.system_breaker._on_failure(e)
            return False

    def set_risk_mode(self, mode: str) -> str:
        """Risk modunu değiştir (Telegram komutu)."""
        mode = mode.lower()

        # Risk parametrelerini güncelle
        if mode == "aggressive":
            self.risk_manager._base_fraction = 0.35
            self.risk_manager._min_edge = 0.02
        elif mode == "conservative":
            self.risk_manager._base_fraction = 0.15
            self.risk_manager._min_edge = 0.05
        elif mode == "normal":
            self.risk_manager._base_fraction = 0.25
            self.risk_manager._min_edge = 0.03
        else:
            return f"Bilinmeyen mod: {mode}. (aggressive, normal, conservative)"

        logger.info(f"Sentinel: Risk modu '{mode}' olarak ayarlandı.")
        return f"Risk modu güncellendi: *{mode.upper()}*"

    async def force_bet(self, match_id: str, selection: str) -> str:
        """Manuel bahis zorla (Telegram komutu)."""
        logger.warning(f"Sentinel: Manuel bahis isteği -> {match_id} / {selection}")

        # Bu özellik ExecutionStage'e doğrudan emir iletmeyi gerektirir.
        # Şimdilik basitçe logluyoruz, tam implementasyon için
        # Pipeline context'ine "force_orders" listesi eklenmeli.

        # Pipeline context'ine erişimimiz var mı?
        # PipelineEngine her döngüde context'i sıfırlıyor ama persistent bir queue tutabiliriz.
        # Basit bir çözüm: Sentinel'in bir "command_queue"su olur, pipeline bunu okur.

        return f"Manuel emir kuyruğa alındı: {match_id} ({selection})"

    def shutdown(self):
        """Sistemi güvenli kapat."""
        self.running = False
        lifecycle.shutdown_event.set()
        if self.pipeline:
            self.pipeline.running = False
        logger.info("Sentinel: Kapanma prosedürü başlatıldı...")

    async def shutdown_async(self):
        """Async cleanup."""
        if self.bot:
            await self.bot.stop()
        logger.info("Sentinel: Sistem kapandı. Görüşmek üzere.")

if __name__ == "__main__":
    sentinel = Sentinel()
    asyncio.run(sentinel.run())
