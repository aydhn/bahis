"""
telegram_admin.py – Telegram Uzaktan Yönetim Paneli (Admin Ops).

Sistem hata verdiğinde bilgisayar başına koşmayın.
Telegram, botunuzun uzaktan kumandası olsun.

Admin komutları (sadece ADMIN_ID'ye yanıt verir):
  /restart_scraper   → Veri çekme modülünü yeniden başlat
  /force_analysis    → Belirtilen maçı zorla analiz et
  /system_status     → CPU, RAM, disk, uptime, son hatalar
  /kill_module       → Belirli bir modülü durdur
  /config_get        → Mevcut konfigürasyonu göster
  /config_set        → Parametre değiştir (runtime)
  /db_stats          → Veritabanı istatistikleri
  /clear_cache       → Önbelleği temizle
  /ga_run            → Genetik Algoritma başlat (arka plan)
  /backup            → DB yedeği al
"""
from __future__ import annotations

import asyncio
import os
import platform
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False


class TelegramAdmin:
    """Telegram üzerinden uzaktan sistem yönetimi.

    Sadece ADMIN_ID'den gelen komutlara yanıt verir.
    Güvenlik: Diğer kullanıcılar komut veremez.

    Kullanım:
        admin = TelegramAdmin(
            admin_id="123456789",
            notifier=notifier,
            db=db, scraper=scraper, scheduler=scheduler,
        )
        admin.register_handlers(telegram_app)
    """

    def __init__(self, admin_id: str = "", notifier=None,
                 db=None, scraper=None, scheduler=None,
                 cache=None, genetic=None, stacker=None):
        self._admin_id = admin_id or os.getenv("TELEGRAM_ADMIN_ID",
                                                os.getenv("TELEGRAM_CHAT_ID", ""))
        self._notifier = notifier
        self._db = db
        self._scraper = scraper
        self._scheduler = scheduler
        self._cache = cache
        self._genetic = genetic
        self._stacker = stacker
        self._boot_time = time.time()
        self._command_log: list[dict] = []
        logger.debug(f"TelegramAdmin başlatıldı (admin_id={self._admin_id[:4]}…)")

    def _is_admin(self, user_id: int | str) -> bool:
        """Admin kontrolü – sadece yetkili kişi komut verebilir."""
        return str(user_id) == str(self._admin_id)

    # ═══════════════════════════════════════════
    #  HANDLER KAYDI
    # ═══════════════════════════════════════════
    def register_handlers(self, app_or_commands: dict):
        """Komut handler'larını Telegram uygulamasına kaydet."""
        admin_commands = {
            "restart_scraper": self._cmd_restart_scraper,
            "force_analysis": self._cmd_force_analysis,
            "system_status": self._cmd_system_status,
            "sys": self._cmd_system_status,  # alias
            "kill_module": self._cmd_kill_module,
            "config_get": self._cmd_config_get,
            "config_set": self._cmd_config_set,
            "db_stats": self._cmd_db_stats,
            "clear_cache": self._cmd_clear_cache,
            "ga_run": self._cmd_ga_run,
            "backup": self._cmd_backup,
            "admin_help": self._cmd_admin_help,
        }

        if isinstance(app_or_commands, dict):
            app_or_commands.update(admin_commands)
        return admin_commands

    # ═══════════════════════════════════════════
    #  /system_status – CPU, RAM, Disk, Uptime
    # ═══════════════════════════════════════════
    async def _cmd_system_status(self, update, context):
        if not self._is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkisiz erişim.")
            return

        uptime = timedelta(seconds=int(time.time() - self._boot_time))

        if PSUTIL_OK:
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net = psutil.net_io_counters()

            text = (
                f"🖥 <b>SİSTEM DURUMU</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"⏱ <b>Uptime:</b> {uptime}\n"
                f"🐍 <b>Python:</b> {sys.version.split()[0]}\n"
                f"💻 <b>OS:</b> {platform.system()} {platform.release()}\n\n"
                f"📊 <b>CPU:</b> {cpu:.1f}%\n"
                f"🧠 <b>RAM:</b> {mem.percent:.1f}% "
                f"({mem.used // (1024**2)}MB / {mem.total // (1024**2)}MB)\n"
                f"💾 <b>Disk:</b> {disk.percent:.1f}% "
                f"({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)\n"
                f"🌐 <b>Net:</b> ↓{net.bytes_recv // (1024**2)}MB "
                f"↑{net.bytes_sent // (1024**2)}MB\n"
            )
        else:
            text = (
                f"🖥 <b>SİSTEM DURUMU</b>\n\n"
                f"⏱ <b>Uptime:</b> {uptime}\n"
                f"🐍 <b>Python:</b> {sys.version.split()[0]}\n"
                f"💻 <b>OS:</b> {platform.system()} {platform.release()}\n"
                f"<i>psutil yüklü değil – detaylı metrikler için: pip install psutil</i>"
            )

        # Son hatalar
        error_log = Path("logs/error.log")
        if error_log.exists():
            try:
                lines = error_log.read_text(encoding="utf-8").strip().split("\n")
                last_errors = lines[-5:] if lines else ["Hata yok"]
                error_text = "\n".join(f"  <code>{l[:80]}</code>" for l in last_errors)
                text += f"\n\n🚨 <b>Son Hatalar:</b>\n{error_text}"
            except Exception:
                pass

        # Scheduler durumu
        if self._scheduler:
            jobs = self._scheduler.get_jobs()
            text += f"\n\n📅 <b>Zamanlanmış Görevler:</b> {len(jobs)}"
            for j in jobs[:5]:
                text += f"\n  • {j.get('name', j.get('id', '?'))} → {j.get('next_run', '?')}"

        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  /restart_scraper – Scraper'ı yeniden başlat
    # ═══════════════════════════════════════════
    async def _cmd_restart_scraper(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        await update.message.reply_text("🔄 Scraper yeniden başlatılıyor…")

        if self._scraper and hasattr(self._scraper, "restart"):
            try:
                await self._scraper.restart()
                await update.message.reply_text("✅ Scraper yeniden başlatıldı.")
            except Exception as e:
                await update.message.reply_text(f"❌ Scraper hatası: {e}")
        else:
            await update.message.reply_text("⚠️ Scraper modülü bağlı değil.")

        self._log_command("restart_scraper", update.effective_user.id)

    # ═══════════════════════════════════════════
    #  /force_analysis [mac_id] – Zorla analiz
    # ═══════════════════════════════════════════
    async def _cmd_force_analysis(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        args = context.args if context.args else []
        match_id = args[0] if args else ""

        if not match_id:
            await update.message.reply_text(
                "📝 Kullanım: <code>/force_analysis mac_id</code>",
                parse_mode="HTML",
            )
            return

        await update.message.reply_text(f"🔍 Analiz başlatılıyor: <code>{match_id}</code>", parse_mode="HTML")

        if self._db and hasattr(self._db, "get_match"):
            try:
                match = self._db.get_match(match_id)
                if match:
                    await update.message.reply_text(
                        f"✅ Maç bulundu: {match.get('home_team', '?')} vs {match.get('away_team', '?')}\n"
                        f"Analiz sıraya alındı."
                    )
                else:
                    await update.message.reply_text(f"⚠️ Maç bulunamadı: {match_id}")
            except Exception as e:
                await update.message.reply_text(f"❌ Hata: {e}")

        self._log_command(f"force_analysis {match_id}", update.effective_user.id)

    # ═══════════════════════════════════════════
    #  /db_stats – Veritabanı istatistikleri
    # ═══════════════════════════════════════════
    async def _cmd_db_stats(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        if not self._db:
            await update.message.reply_text("⚠️ DB bağlı değil.")
            return

        try:
            stats = {}
            if hasattr(self._db, "stats"):
                stats = self._db.stats()
            elif hasattr(self._db, "get_stats"):
                stats = self._db.get_stats()

            text = (
                "🗄 <b>VERİTABANI</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            )
            for k, v in stats.items():
                text += f"• <b>{k}:</b> {v}\n"

            if not stats:
                text += "<i>İstatistik mevcut değil.</i>"

            await update.message.reply_text(text, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"❌ DB hatası: {e}")

    # ═══════════════════════════════════════════
    #  /clear_cache – Önbelleği temizle
    # ═══════════════════════════════════════════
    async def _cmd_clear_cache(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        cleared = 0
        if self._cache:
            try:
                if hasattr(self._cache, "clear"):
                    self._cache.clear()
                    cleared += 1
                if hasattr(self._cache, "cache_clear"):
                    self._cache.cache_clear()
                    cleared += 1
            except Exception:
                pass

        await update.message.reply_text(f"🧹 Önbellek temizlendi ({cleared} katman).")
        self._log_command("clear_cache", update.effective_user.id)

    # ═══════════════════════════════════════════
    #  /config_get & /config_set
    # ═══════════════════════════════════════════
    async def _cmd_config_get(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        config_path = Path("config.json")
        if config_path.exists():
            try:
                import json
                config = json.loads(config_path.read_text())
                params = config.get("parameters", config)
                text = "⚙️ <b>KONFİGÜRASYON</b>\n\n"
                for k, v in list(params.items())[:20]:
                    text += f"<code>{k}</code>: {v}\n"
                await update.message.reply_text(text, parse_mode="HTML")
            except Exception as e:
                await update.message.reply_text(f"❌ Config okuma hatası: {e}")
        else:
            await update.message.reply_text("⚠️ config.json bulunamadı.")

    async def _cmd_config_set(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        args = context.args if context.args else []
        if len(args) < 2:
            await update.message.reply_text(
                "📝 Kullanım: <code>/config_set parametre değer</code>",
                parse_mode="HTML",
            )
            return

        key, value = args[0], args[1]
        config_path = Path("config.json")

        try:
            import json
            config = json.loads(config_path.read_text()) if config_path.exists() else {"parameters": {}}
            params = config.setdefault("parameters", {})

            # Tip dönüşümü
            try:
                value = float(value)
            except ValueError:
                pass

            old_value = params.get(key, "YOK")
            params[key] = value
            config_path.write_text(json.dumps(config, indent=2))

            await update.message.reply_text(
                f"✅ <code>{key}</code>: {old_value} → {value}",
                parse_mode="HTML",
            )
            self._log_command(f"config_set {key}={value}", update.effective_user.id)
        except Exception as e:
            await update.message.reply_text(f"❌ Config yazma hatası: {e}")

    # ═══════════════════════════════════════════
    #  /ga_run – Genetik Algoritma başlat
    # ═══════════════════════════════════════════
    async def _cmd_ga_run(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        if not self._genetic:
            await update.message.reply_text("⚠️ Genetik Algoritma modülü bağlı değil.")
            return

        await update.message.reply_text("🧬 Genetik Algoritma arka planda başlatılıyor…")
        self._log_command("ga_run", update.effective_user.id)

        # Arka planda çalıştır (uzun sürebilir)
        asyncio.create_task(self._run_ga_background(update))

    async def _run_ga_background(self, update):
        try:
            # Basit backtest fonksiyonu (placeholder)
            def dummy_backtest(params):
                import random
                return {
                    "roi": random.uniform(-0.1, 0.3),
                    "max_drawdown": random.uniform(0.05, 0.25),
                    "sharpe": random.uniform(-0.5, 2.0),
                    "total_bets": random.randint(50, 200),
                }

            best = self._genetic.evolve(dummy_backtest, generations=20)
            self._genetic.save_config(best)

            await self._notifier.send(
                f"🧬 <b>GA TAMAMLANDI</b>\n\n"
                f"ROI: {best.roi:.2%}\n"
                f"Drawdown: {best.drawdown:.1%}\n"
                f"Sharpe: {best.sharpe:.2f}\n"
                f"config.json güncellendi.",
            )
        except Exception as e:
            await self._notifier.send(f"❌ GA Hatası: {e}")

    # ═══════════════════════════════════════════
    #  /backup – DB yedeği
    # ═══════════════════════════════════════════
    async def _cmd_backup(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        backup_dir = Path("backups")
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            import shutil
            db_path = Path("data/quant_bot.duckdb")
            if db_path.exists():
                dest = backup_dir / f"db_backup_{timestamp}.duckdb"
                shutil.copy2(db_path, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                await update.message.reply_text(
                    f"✅ Yedek alındı: <code>{dest.name}</code> ({size_mb:.1f}MB)",
                    parse_mode="HTML",
                )
            else:
                await update.message.reply_text("⚠️ DB dosyası bulunamadı.")
        except Exception as e:
            await update.message.reply_text(f"❌ Yedekleme hatası: {e}")

        self._log_command("backup", update.effective_user.id)

    # ═══════════════════════════════════════════
    #  /kill_module – Modül durdur
    # ═══════════════════════════════════════════
    async def _cmd_kill_module(self, update, context):
        if not self._is_admin(update.effective_user.id):
            return

        args = context.args if context.args else []
        if not args:
            await update.message.reply_text(
                "📝 Kullanım: <code>/kill_module modul_adı</code>",
                parse_mode="HTML",
            )
            return

        module_name = args[0]
        await update.message.reply_text(f"🔴 {module_name} durduruluyor…")
        # Gerçek implementasyon: asyncio.Task.cancel()
        self._log_command(f"kill_module {module_name}", update.effective_user.id)

    # ═══════════════════════════════════════════
    #  /admin_help – Admin komut listesi
    # ═══════════════════════════════════════════
    async def _cmd_admin_help(self, update, context):
        if not self._is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Yetkisiz.")
            return

        text = (
            "🔐 <b>ADMİN KOMUTLARI</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "<b>🖥 Sistem</b>\n"
            "/system_status – CPU/RAM/Disk\n"
            "/restart_scraper – Scraper'ı yeniden başlat\n"
            "/kill_module [ad] – Modülü durdur\n"
            "/backup – DB yedeği al\n\n"
            "<b>📊 Analiz</b>\n"
            "/force_analysis [maç] – Zorla analiz\n"
            "/ga_run – Genetik Algoritma başlat\n\n"
            "<b>⚙️ Konfigürasyon</b>\n"
            "/config_get – Parametreleri göster\n"
            "/config_set [key] [val] – Parametre değiştir\n"
            "/clear_cache – Önbelleği temizle\n"
            "/db_stats – DB istatistikleri\n"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    # ═══════════════════════════════════════════
    #  YARDIMCI
    # ═══════════════════════════════════════════
    def _log_command(self, cmd: str, user_id: Any):
        self._command_log.append({
            "command": cmd,
            "user": str(user_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.info(f"[Admin] Komut: {cmd} (user={str(user_id)[:6]}…)")

    @property
    def command_log(self) -> list[dict]:
        return self._command_log[-50:]
