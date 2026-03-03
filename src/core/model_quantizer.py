"""
model_quantizer.py – Model sıkıştırma (INT8 / FP16).
Büyük modelleri küçülterek RAM kullanımını düşürür.
"""
from __future__ import annotations


import numpy as np
from loguru import logger


class ModelQuantizer:
    """Model boyutunu ve RAM kullanımını optimize eder."""

    def __init__(self):
        self._original_sizes: dict[str, int] = {}
        self._quantized_sizes: dict[str, int] = {}
        logger.debug("ModelQuantizer başlatıldı.")

    def quantize_numpy(self, array: np.ndarray, dtype: str = "float16") -> np.ndarray:
        """Numpy dizisini düşük hassasiyete dönüştürür."""
        original_bytes = array.nbytes
        target_dtype = getattr(np, dtype, np.float16)
        quantized = array.astype(target_dtype)
        new_bytes = quantized.nbytes
        ratio = (1 - new_bytes / max(original_bytes, 1)) * 100
        logger.debug(f"Quantize: {original_bytes/1e6:.1f}MB → {new_bytes/1e6:.1f}MB ({ratio:.0f}% tasarruf)")
        return quantized

    def quantize_torch_model(self, model, method: str = "dynamic"):
        """PyTorch modelini quantize eder."""
        try:
            import torch
            import torch.quantization as quant

            original_size = sum(p.nelement() * p.element_size() for p in model.parameters())
            self._original_sizes["torch_model"] = original_size

            if method == "dynamic":
                quantized = quant.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            elif method == "fp16":
                quantized = model.half()
            else:
                quantized = model

            new_size = sum(p.nelement() * p.element_size() for p in quantized.parameters())
            self._quantized_sizes["torch_model"] = new_size

            ratio = (1 - new_size / max(original_size, 1)) * 100
            logger.info(f"Model quantize edildi: {original_size/1e6:.1f}MB → {new_size/1e6:.1f}MB ({ratio:.0f}%)")
            return quantized

        except ImportError:
            logger.warning("PyTorch yüklü değil – quantization atlanıyor.")
            return model
        except Exception as e:
            logger.error(f"Quantization hatası: {e}")
            return model

    def estimate_memory(self, model_params: int, dtype: str = "float32") -> dict:
        """Model bellek kullanımını tahmin eder."""
        bytes_per_param = {"float32": 4, "float16": 2, "int8": 1, "int4": 0.5}
        bpp = bytes_per_param.get(dtype, 4)
        memory_mb = model_params * bpp / 1e6

        return {
            "params": model_params,
            "dtype": dtype,
            "memory_mb": memory_mb,
            "memory_gb": memory_mb / 1024,
        }

    def savings_report(self) -> dict:
        total_original = sum(self._original_sizes.values())
        total_quantized = sum(self._quantized_sizes.values())
        saved = total_original - total_quantized

        return {
            "original_mb": total_original / 1e6,
            "quantized_mb": total_quantized / 1e6,
            "saved_mb": saved / 1e6,
            "compression_ratio": total_quantized / max(total_original, 1),
        }
