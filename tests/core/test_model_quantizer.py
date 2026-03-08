import pytest
from unittest.mock import MagicMock, patch
import sys
import numpy as np
from src.core.model_quantizer import ModelQuantizer

class MockParameter:
    def __init__(self, elements=1000, element_size=4):
        self._elements = elements
        self._element_size = element_size

    def nelement(self):
        return self._elements

    def element_size(self):
        return self._element_size

class MockModel:
    def __init__(self, fail_half=False):
        self.fail_half = fail_half
        self.params = [MockParameter(1000, 4), MockParameter(2000, 4)]
        self.half_model = MagicMock()
        self.half_model.parameters.return_value = [MockParameter(1000, 2), MockParameter(2000, 2)]

    def parameters(self):
        return self.params

    def half(self):
        if self.fail_half:
            raise RuntimeError("Simulation of .half() failure")
        return self.half_model

def test_quantize_numpy():
    quantizer = ModelQuantizer()
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    quantized = quantizer.quantize_numpy(array, dtype="float16")

    assert quantized.dtype == np.float16
    assert np.allclose(array, quantized)

@patch.dict(sys.modules, {'torch': MagicMock(), 'torch.quantization': MagicMock()})
def test_quantize_torch_model_dynamic():
    import torch
    import torch.quantization as quant

    torch.nn.Linear = "LinearLayer"
    torch.qint8 = "qint8"

    # Mock quantize_dynamic to return a smaller model
    mock_quantized_model = MockModel()
    mock_quantized_model.params = [MockParameter(1000, 1), MockParameter(2000, 1)]
    quant.quantize_dynamic.return_value = mock_quantized_model

    quantizer = ModelQuantizer()
    model = MockModel()

    result = quantizer.quantize_torch_model(model, method="dynamic")

    # Verify quantize_dynamic was called correctly
    quant.quantize_dynamic.assert_called_once_with(
        model, {"LinearLayer"}, dtype="qint8"
    )

    assert result == mock_quantized_model

    original_size = (1000 * 4) + (2000 * 4)
    quantized_size = (1000 * 1) + (2000 * 1)

    assert quantizer._original_sizes["torch_model"] == original_size
    assert quantizer._quantized_sizes["torch_model"] == quantized_size

@patch.dict(sys.modules, {'torch': MagicMock(), 'torch.quantization': MagicMock()})
def test_quantize_torch_model_fp16():
    quantizer = ModelQuantizer()
    model = MockModel()

    result = quantizer.quantize_torch_model(model, method="fp16")

    assert result == model.half_model

    original_size = (1000 * 4) + (2000 * 4)
    quantized_size = (1000 * 2) + (2000 * 2)

    assert quantizer._original_sizes["torch_model"] == original_size
    assert quantizer._quantized_sizes["torch_model"] == quantized_size

@patch.dict(sys.modules, {'torch': MagicMock(), 'torch.quantization': MagicMock()})
def test_quantize_torch_model_unknown_method():
    quantizer = ModelQuantizer()
    model = MockModel()

    result = quantizer.quantize_torch_model(model, method="unknown")

    # Returns original model if method is unknown
    assert result == model

    original_size = (1000 * 4) + (2000 * 4)

    assert quantizer._original_sizes["torch_model"] == original_size
    assert quantizer._quantized_sizes["torch_model"] == original_size

def test_quantize_torch_model_import_error():
    # Simulate missing torch
    with patch.dict(sys.modules, {'torch': None}):
        quantizer = ModelQuantizer()
        model = MockModel()

        result = quantizer.quantize_torch_model(model, method="dynamic")

        assert result == model
        assert "torch_model" not in quantizer._original_sizes

@patch.dict(sys.modules, {'torch': MagicMock(), 'torch.quantization': MagicMock()})
def test_quantize_torch_model_exception():
    quantizer = ModelQuantizer()
    model = MockModel(fail_half=True)

    result = quantizer.quantize_torch_model(model, method="fp16")

    assert result == model
    # Original size should be recorded before exception
    original_size = (1000 * 4) + (2000 * 4)
    assert quantizer._original_sizes["torch_model"] == original_size
    # Quantized size should not be recorded
    assert "torch_model" not in quantizer._quantized_sizes
