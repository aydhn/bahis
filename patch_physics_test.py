import re
with open("tests/test_physics_engines.py", "r") as f:
    content = f.read()

# Fix path signature assert (DataFrame from pl is not using is_empty on dict rows)
# Wait, polars DataFrame does have is_empty()
# Let's see why it failed. `AssertionError: True is not false` means res is empty.
content = content.replace("self.assertFalse(res.is_empty())", "self.assertTrue(len(res) > 0 if not hasattr(res, 'is_empty') else not res.is_empty())")

# Wait, res might be completely empty because extract returns empty DataFrame if results is empty.
# In `extract`: `return pl.DataFrame(results) if results else pl.DataFrame()`
# Why would `results` be empty? Because `for row in features.iter_rows(named=True):` might be empty.
# `features` is `df`. `df` has 1 row.
# Let's check `test_path_signature` again.
# df = pl.DataFrame({"match_id": ["m1"], ...})
# Wait, `polars.DataFrame.iter_rows(named=True)` yields dictionaries? No, it yields dict-like? No, polars `iter_rows(named=True)` usually returns `dict` or `namedtuple`. Wait, named=True returns dict.
