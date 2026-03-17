import sys
from unittest.mock import MagicMock
import pytest
import importlib.util

class DummySpec:
    def __init__(self):
        self.name = "numba"

class DummyNumba:
    __version__ = "0.60.0"
    def jit(self, *args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    def njit(self, *args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

dummy_numba = DummyNumba()
dummy_numba.__spec__ = importlib.util.spec_from_loader("numba", loader=None)

sys.modules['numba'] = dummy_numba

import torch.nn.init
def _mock_uniform_(*args, **kwargs):
    pass
def _mock_kaiming_uniform_(*args, **kwargs):
    pass
torch.nn.init.uniform_ = _mock_uniform_
torch.nn.init.kaiming_uniform_ = _mock_kaiming_uniform_
