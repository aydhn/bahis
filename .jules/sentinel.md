# Sentinel's Journal

## 2026-02-23 - Hardcoded Neo4j Password
**Vulnerability:** A hardcoded password ("bahis_graph_2026") was used as a fallback for the Neo4j database connection in `src/memory/neo4j_graph.py`.
**Learning:** The fallback mechanism was likely intended for developer convenience or a specific deployment, but it exposed a secret directly in the source code.
**Prevention:** Avoid default values for sensitive credentials in code. Use environment variables strictly, and fail or log a warning if they are missing.

## 2024-03-04 - DuckDB read_only Regex Bypass
**Vulnerability:** SQL Injection/Arbitrary File Read bypass in text-to-SQL logic using DuckDB. The blocklist relied on word boundary regex `\b(READ_CSV)\b`, which failed to block `read_csv_auto` or `read_text` because the underscore allowed it to bypass the boundary or wasn't even listed.
**Learning:** Blocklists that rely on exact string matching or rigid word boundaries can be easily bypassed if the underlying SQL dialect has flexible or extensive variations of dangerous functions (e.g., DuckDB's `read_*` functions allowing arbitrary file reading even in `read_only=True` mode).
**Prevention:** Instead of blocking exact function names, block prefixes using `\bprefix_\w*\b` (like `\bread_\w*\b` for DuckDB) to robustly match all variations.
