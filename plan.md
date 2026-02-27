# Plan to Upgrade the Quantitative Betting System

**Objective:** Implement massive upgrades maximizing execution speed, minimizing error (Zero Error), and achieving high real profitability. Integrating philosophies of Google/Binance CEOs (tech/crypto execution), JP Morgan Risk (solvency), Bill Benter (quant model), and a Senior Quant Dev. Replacing paid services with local/free alternatives.

## Proposed Changes

1.  **Enhance LLM/RAG Dependency with Local Open-Source Models (Ollama)**
    -   *Action:* Modify `src/ingestion/news_rag.py` to add a new `_analyze_ollama` method for local sentiment analysis using an Ollama endpoint (defaulting to `http://localhost:11434/api/generate` and `llama3` model). Add a fallback sequence: Ollama -> Gemini -> HF -> Rule-based. Modify `__init__` to accept `ollama_url` and `ollama_model`.
    -   *Verification:* Use `read_file` on `src/ingestion/news_rag.py` to ensure the new method and fallback logic are correctly implemented.

2.  **Optimize the Benter Model (Adding Advanced Edge Detection & Fractional Kelly)**
    -   *Action:* Modify `src/quant/models/benter_model.py`. Add a `fractional_kelly` multiplier parameter (default 0.5 for Half-Kelly, matching real-world bankroll safety) to the `detect_value_bet` method. Update the returned dictionaries to include a `suggested_stake_fraction` calculated as `(edge / (o - 1)) * fractional_kelly`. Add logic to adjust `min_edge` dynamically based on model uncertainty if provided in the outcomes.
    -   *Verification:* Use `read_file` on `src/quant/models/benter_model.py` to confirm the Kelly multiplier logic is properly integrated.

3.  **Upgrade the Ensemble Model for Ultra-Fast Execution & Intelligent Pruning**
    -   *Action:* Modify `src/quant/models/ensemble.py`. In `predict`, use `asyncio.gather` (via `asyncio.to_thread` for synchronous sub-models) to run all active sub-models in parallel rather than a blocking `for` loop. Implement logic in the pruning section: if `model_health[name]["errors"] >= 3` or if a dynamic weight drops below `0.05`, temporarily disable the model (`status="PRUNED"`) and skip running it entirely for speed.
    -   *Verification:* Use `read_file` on `src/quant/models/ensemble.py` to verify the async execution and pruning logic.

4.  **Refactor Inference Stage for Maximum Throughput**
    -   *Action:* Modify `src/pipeline/stages/inference.py`. Ensure the `_analyze_single_match` tasks are explicitly gathered with exception handling that logs the error but does not fail the batch. Add an LRU cache (using `functools.lru_cache` or a simple dict) for the `SimilarityEngine` lookups inside `_analyze_single_match` to prevent duplicate DB/computation for similar odds.
    -   *Verification:* Use `read_file` on `src/pipeline/stages/inference.py` to check the `asyncio.gather` implementation and caching logic.

5.  **Run System Verification Tests**
    -   *Action:* Run `python tests/verify_pipeline.py` or equivalent test scripts using `run_in_bash_session` to ensure no regressions were introduced by the changes to the models and pipeline.

6.  **Pre-commit checks**
    -   *Action:* Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.

7.  **Submit changes**
    -   *Action:* Submit the branch with all the new updates.
