## 2024-05-19 - Optimization Targets
**Learning:** Found potential optimization in InferenceStage. `_similarity_cache` key is created dynamically per row. MultiTaskBackbone inference may fall back to Polars DataFrame creation if `features.is_empty()` is false but `mtl_predictions` is empty. The `_analyze_single_match` method is called concurrently for every match, which could be slow. `EnsembleModel.predict` spins up threads for each sub-model, even though `_analyze_single_match` already spins up a thread. This means O(N*M) threads where N is matches and M is models, potentially causing thread thrashing.

**Action:** Look into thread pooling or batched predictions for `EnsembleModel` or `InferenceStage`.

## 2024-05-19 - `_similarity_cache` Optimization
**Learning:** `_similarity_cache` has a flaw. When it reaches 1000 items, it is cleared completely (`self._similarity_cache.clear()`). This means all cache entries are dropped at once, causing a severe drop in hit rate and potentially leading to a thundering herd problem where many requests suddenly miss the cache and do expensive calculations. It would be better to implement a simple LRU cache or at least not clear the whole dictionary when it gets full, using python's `functools.lru_cache` or a simple LRU dict implementation.
**Action:** Replace `_similarity_cache` dictionary management with Python's standard `functools.lru_cache` to handle evictions efficiently and naturally without explicitly clearing the whole dict.

## 2024-05-19 - `_similarity_cache` Optimization
**Learning:** `_similarity_cache` has a flaw. When it reaches 1000 items, it is cleared completely (`self._similarity_cache.clear()`). This means all cache entries are dropped at once, causing a severe drop in hit rate and potentially leading to a thundering herd problem where many requests suddenly miss the cache and do expensive calculations. Also, applying `@lru_cache` directly to a class method caches the `self` argument as part of the key. This prevents the `InferenceStage` instance from being garbage-collected until the cache is evicted or cleared, which can cause memory leaks depending on how often the class is instantiated.
**Action:** Replace `_similarity_cache` dictionary management with Python's standard `functools.lru_cache` to handle evictions efficiently. To avoid memory leaks with `self`, decouple the cache logic from the class method or be aware of the lifecycle of `InferenceStage` (it's often a singleton in the pipeline).

## 2026-03-01 - lru_cache Memory Leak on Class Methods
**Learning:** Using `@lru_cache` directly on a class method caches the `self` argument, which keeps the instance alive and prevents garbage collection, leading to memory leaks. This is especially problematic if the class is instantiated multiple times.
**Action:** Apply `lru_cache` to instance methods inside `__init__` instead of using the decorator on the class method definition (e.g., `self.method = lru_cache(maxsize=...)(self._method_impl)`).

## $(date +%Y-%m-%d) - O(1) Cache Eviction using Python 3.7+ Dicts
**Learning:** LRU and TTL caches tracking oldest entries via `min(dict, key=...)` create an O(N) performance bottleneck for large caches. In Python 3.7+, dictionaries strictly maintain insertion order, meaning the oldest un-evicted item is always the first one `next(iter(dict))`.
**Action:** Always utilize dictionary insertion order for caches in modern Python. Delete and re-insert items to refresh their TTL/LRU position to the end. For max size eviction, use `del dict[next(iter(dict))]` for O(1) performance instead of O(N) `min()` scans.

## 2026-03-03 - O(1) Dictionary Caching for N+1 Query Elimination in DBManager
**Learning:** `DBManager.build_feature_matrix` was executing 3 independent queries per row (`self.get_team_stats()` and `self.get_odds_history()`) which scales as O(N) when there are N matches. Doing an upfront batch query `SELECT ... IN (?,?)` and mapping it to a dictionary allows $O(1)$ memory access without making per-row DB calls.
**Action:** Always batch load related tabular data prior to iterating rows when assembling feature matrices.

## 2024-03-03 - [Telegram Bot] Prevent Event Loop Blocking During I/O
**Learning:** Synchronous file reads (like `json.load(open(path))`) within an `async def` function block the event loop, causing massive latency spikes (~470ms under load) for all concurrent tasks.
**Action:** Always wrap synchronous file I/O operations inside `asyncio.to_thread` when executing within async contexts, rather than relying on external libraries like `aiofiles` unless already installed.

## 2026-03-03 - Nested Thread Pool Thrashing in Concurrent Async Contexts
**Learning:** `EnsembleModel.predict` was using a `ThreadPoolExecutor` internally to parallelize running sub-models. However, this method is typically called inside `asyncio.to_thread` per-match (which means N matches x M models = N*M threads concurrently scaling up). The internal pool leads to heavy thread creation overhead and thrashing, reducing overall throughput.
**Action:** Do not use nested thread pools when the outer execution loop is already concurrent/parallelized. Execute sub-tasks sequentially in the inner loop instead.
