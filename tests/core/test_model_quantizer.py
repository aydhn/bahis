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

def test_savings_report():
    quantizer = ModelQuantizer()
    quantizer._original_sizes = {"model1": 10000000, "model2": 5000000}
    quantizer._quantized_sizes = {"model1": 2500000, "model2": 1250000}

    report = quantizer.savings_report()

    assert report["original_mb"] == 15.0
    assert report["quantized_mb"] == 3.75
    assert report["saved_mb"] == 11.25
    assert report["compression_ratio"] == 3750000 / 15000000

def test_savings_report_empty():
    quantizer = ModelQuantizer()
    report = quantizer.savings_report()

    assert report["original_mb"] == 0.0
    assert report["quantized_mb"] == 0.0
    assert report["saved_mb"] == 0.0
    assert report["compression_ratio"] == 0.0

def test_estimate_memory():
    quantizer = ModelQuantizer()

    # Test float32 (default)
    result = quantizer.estimate_memory(1000000, dtype="float32")
    assert result["params"] == 1000000
    assert result["dtype"] == "float32"
    assert result["memory_mb"] == 4.0
    assert result["memory_gb"] == 4.0 / 1024

    # Test float16
    result = quantizer.estimate_memory(1000000, dtype="float16")
    assert result["memory_mb"] == 2.0

    # Test int8
    result = quantizer.estimate_memory(1000000, dtype="int8")
    assert result["memory_mb"] == 1.0

    # Test int4
    result = quantizer.estimate_memory(1000000, dtype="int4")
    assert result["memory_mb"] == 0.5

    # Test unknown dtype falls back to default 4 bytes
    result = quantizer.estimate_memory(1000000, dtype="unknown")
    assert result["memory_mb"] == 4.0

def test_quantize_numpy_int8():
    quantizer = ModelQuantizer()
    array = np.array([1, 2, 3], dtype=np.int32)
    quantized = quantizer.quantize_numpy(array, dtype="int8")

    assert quantized.dtype == np.int8
    assert np.array_equal(array, quantized)

def test_quantize_numpy_fallback():
    quantizer = ModelQuantizer()
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # Provide an unknown dtype, should fallback to float16
    quantized = quantizer.quantize_numpy(array, dtype="unknown_dtype")

    assert quantized.dtype == np.float16
    assert np.allclose(array, quantized)

def test_quantize_numpy_empty_array():
    quantizer = ModelQuantizer()
    array = np.array([], dtype=np.float32)
    quantized = quantizer.quantize_numpy(array, dtype="float16")

    assert quantized.dtype == np.float16
    assert len(quantized) == 0

@patch("src.core.model_quantizer.logger")
def test_quantize_numpy_logging(mock_logger):
    quantizer = ModelQuantizer()
    # 1000 float32 elements = 4000 bytes
    array = np.ones(1000, dtype=np.float32)
    quantized = quantizer.quantize_numpy(array, dtype="float16")

    # Check if logger.debug was called to report savings
    mock_logger.debug.assert_called()
    # 4000 bytes to 2000 bytes -> 50% savings
    # original_bytes/1e6 = 0.0MB, new_bytes/1e6 = 0.0MB
    args, kwargs = mock_logger.debug.call_args
    assert "Quantize" in args[0]
    assert "50% tasarruf" in args[0]

def test_estimate_memory_edge_cases():
    quantizer = ModelQuantizer()

    # Zero parameters
    result = quantizer.estimate_memory(0, dtype="float32")
    assert result["params"] == 0
    assert result["memory_mb"] == 0.0
    assert result["memory_gb"] == 0.0

    # Negative parameters (should handle mathematically)
    result = quantizer.estimate_memory(-1000, dtype="float32")
    assert result["params"] == -1000
    assert result["memory_mb"] == -0.004
    assert result["memory_gb"] == -0.004 / 1024

    # Extremely large parameter count (e.g. 1 Trillion parameters)
    large_params = 1_000_000_000_000
    result = quantizer.estimate_memory(large_params, dtype="float16")
    assert result["params"] == large_params
    assert result["memory_mb"] == 2_000_000.0
    assert result["memory_gb"] == 2_000_000.0 / 1024

    # Minimal parameter count
    result = quantizer.estimate_memory(1, dtype="int8")
    assert result["params"] == 1
    assert result["memory_mb"] == 1 / 1e6
    assert result["memory_gb"] == (1 / 1e6) / 1024
