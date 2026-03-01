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
