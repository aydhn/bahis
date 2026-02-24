## 2026-02-24 - [Vectorized Score Counting]
**Learning:** Vectorized score counting with numpy.unique on combined integers yielded ~7x speedup (0.15s -> 0.02s) for 100k simulations.
**Action:** Always prefer numpy vectorization for frequency counting on large arrays.
