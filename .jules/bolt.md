## 2026-02-24 - [Vectorized Score Counting]
**Learning:** Vectorized score counting with numpy.unique on combined integers yielded ~7x speedup (0.15s -> 0.02s) for 100k simulations.
**Action:** Always prefer numpy vectorization for frequency counting on large arrays.

## 2026-02-25 - [Bulk Monte Carlo Simulation]
**Learning:** Vectorizing Monte Carlo simulations for 1000 matches (10k sims each) yielded ~1.5x speedup (2.0s -> 1.3s). The cost is dominated by `np.random.poisson` generation (10M samples), so loop overhead was about 35% of the total time.
**Action:** For heavy simulation tasks, batch processing is faster, but `numpy` random number generation itself has a base cost that scales linearly.
