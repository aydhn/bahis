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

## 2026-02-27 - SQL Injection via String Interpolation in db.query()
**Vulnerability:** `src/system/digital_twin.py` constructed SQL queries by interpolating integer inputs directly into the `SELECT` query statement via Python f-strings. This bypassed the parameterized query support.
**Learning:** Even though the function type hints indicate an integer input, explicitly allowing string concatenation in database queries bypasses any built-in sanitation and introduces potential vector injection vulnerabilities if the types are bypassed anywhere in the calling stack.
**Prevention:** Never use f-strings to pass variables into SQL query strings. Always use parameterized queries and pass parameters using a list (`db.query(sql, [params])`).
## 2024-05-25 - DuckDB Arbitrary File Read bypasses read_only=True
**Vulnerability:** Directly executing LLM-generated DuckDB SQL against the database, even with `read_only=True` mode, allows reading sensitive arbitrary files from the filesystem via functions like `read_csv('/etc/passwd')`.
**Learning:** DuckDB's `read_only=True` mode does not restrict external file access or attaching new databases. It primarily restricts modifications to the attached catalog, but its powerful data ingestion functions can be abused to exfiltrate system files or load remote extensions.
**Prevention:** In addition to parameterized queries or ORMs, when building Text-to-SQL functionality using DuckDB, use strict blocklists for functions like `READ_CSV`, `READ_PARQUET`, `COPY`, `ATTACH`, `INSTALL`, and `LOAD`, or use an isolated AST parser to validate the query strictly to `SELECT` operations within specific schema limits.
