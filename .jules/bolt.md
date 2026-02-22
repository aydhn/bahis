## 2026-02-22 - [Vectorization Impact]
**Learning:** Polars `to_numpy()` might return read-only arrays which breaks in-place operations. Always use `.copy()` or `np.where` when modifying data from Polars.
**Action:** Use `np.where` for data cleaning on numpy arrays derived from Polars.
