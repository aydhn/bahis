## 2026-02-24 - Telegram Bot Authorization Bypass
**Vulnerability:** The Telegram Bot (`src/reporting/telegram_bot.py`) processed commands like `/shutdown` and `/force` from any user who messaged it, without verifying the sender's `chat_id`.
**Learning:** External interfaces (bots, webhooks) must explicitly validate the source identity. Relying on secrecy of the bot handle is insufficient.
**Prevention:** Implemented an allowlist check (`TELEGRAM_ALLOWED_USERS`) in `_handle_update` to reject unauthorized commands. Default to "deny all" if not configured (except legacy admin).
