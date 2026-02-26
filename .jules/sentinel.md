## 2026-02-24 - Telegram Bot Authorization Bypass
**Vulnerability:** The Telegram Bot (`src/reporting/telegram_bot.py`) processed commands like `/shutdown` and `/force` from any user who messaged it, without verifying the sender's `chat_id`.
**Learning:** External interfaces (bots, webhooks) must explicitly validate the source identity. Relying on secrecy of the bot handle is insufficient.
**Prevention:** Implemented an allowlist check (`TELEGRAM_ALLOWED_USERS`) in `_handle_update` to reject unauthorized commands. Default to "deny all" if not configured (except legacy admin).

## 2026-02-25 - Hardcoded Neo4j Credentials
**Vulnerability:** `NEO4J_PASSWORD` was hardcoded with default values "password" and "bahis_graph_2026" in `config.py` and `neo4j_graph.py`.
**Learning:** Fallback values for development convenience can easily become production vulnerabilities. Pydantic's default values make it easy to forget setting env vars.
**Prevention:** Use `Optional[str] = None` for sensitive fields in Pydantic settings to force explicit configuration or graceful degradation, and verify no hardcoded secrets exist in source code.

## 2026-02-26 - SQL Injection via Dictionary Keys in DBManager
**Vulnerability:** `DBManager.upsert_match` constructed SQL queries by directly interpolating dictionary keys into the `INSERT` statement. This allowed SQL injection if the keys were attacker-controlled.
**Learning:** Even internal methods can be dangerous sinks if they assume trusted input structure. Dynamic SQL construction from object keys is a common but subtle vulnerability.
**Prevention:** Whitelist valid column names based on the database schema. Filter all input dictionaries against this whitelist before constructing SQL queries.
