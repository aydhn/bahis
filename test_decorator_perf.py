import sys
from unittest.mock import MagicMock

# Mock loguru to avoid missing module errors
sys.modules['loguru'] = MagicMock()
sys.modules['loguru.logger'] = MagicMock()

import time
import random
from timeit import default_timer as timer

from src.memory.smart_cache import cached

@cached(ttl=3600.0, maxsize=1000)
def compute_value(x):
    return x * 2

def benchmark():
    keys = list(range(100000))
    start = timer()
    for k in keys:
        compute_value(k)
    end = timer()
    print(f"Time taken for cached decorator with 100000 items (O(1) eviction): {end - start:.4f}s")

if __name__ == "__main__":
    benchmark()
