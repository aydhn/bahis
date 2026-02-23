# Sentinel's Journal

## 2026-02-23 - Hardcoded Neo4j Password
**Vulnerability:** A hardcoded password ("bahis_graph_2026") was used as a fallback for the Neo4j database connection in `src/memory/neo4j_graph.py`.
**Learning:** The fallback mechanism was likely intended for developer convenience or a specific deployment, but it exposed a secret directly in the source code.
**Prevention:** Avoid default values for sensitive credentials in code. Use environment variables strictly, and fail or log a warning if they are missing.
