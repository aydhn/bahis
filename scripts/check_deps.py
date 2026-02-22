import sys
try:
    import torch
    print(f"Torch: {torch.__version__}")
except ImportError:
    print("Torch: MISSING")

try:
    import kan
    print(f"KAN: {kan.__version__ if hasattr(kan, '__version__') else 'INSTALLED'}")
except ImportError:
    print("KAN: MISSING")
