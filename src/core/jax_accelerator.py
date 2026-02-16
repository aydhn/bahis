"""
jax_accelerator.py – JAX / XLA ile donanım seviyesinde hızlandırma.
Matematiksel işlemleri derleyerek hızlandırır. JAX yoksa NumPy fallback.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
    logger.info(f"JAX aktif – backend: {jax.default_backend()}")
except ImportError:
    JAX_AVAILABLE = False
    logger.warning("JAX yüklü değil – NumPy fallback kullanılacak.")


class JAXAccelerator:
    """JAX/XLA ile matematiksel operasyonları hızlandırır."""

    def __init__(self):
        self._compiled_fns: dict = {}
        logger.debug("JAXAccelerator başlatıldı.")

    def accelerate(self, features: pl.DataFrame) -> pl.DataFrame:
        """Feature matrisini hızlandırılmış hesaplamalarla zenginleştirir."""
        if features.is_empty():
            return features

        numeric_cols = [c for c in features.columns
                        if features[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]

        if not numeric_cols:
            return features

        mat = features.select(numeric_cols).to_numpy().astype(np.float64)

        if JAX_AVAILABLE:
            enriched = self._jax_compute(mat, numeric_cols)
        else:
            enriched = self._numpy_compute(mat, numeric_cols)

        # Yeni sütunları ekle
        for col_name, values in enriched.items():
            features = features.with_columns(pl.Series(col_name, values))

        return features

    def _jax_compute(self, mat: jnp.ndarray, col_names: list[str]) -> dict:
        """JAX ile derlenmiş hesaplamalar."""
        mat_jax = jnp.array(mat)

        @jit
        def compute_stats(x):
            row_mean = jnp.mean(x, axis=1)
            row_std = jnp.std(x, axis=1)
            row_skew = jnp.mean(((x.T - row_mean) / (row_std + 1e-8)).T ** 3, axis=1)
            row_kurt = jnp.mean(((x.T - row_mean) / (row_std + 1e-8)).T ** 4, axis=1) - 3
            return row_mean, row_std, row_skew, row_kurt

        @jit
        def softmax_rows(x):
            exp_x = jnp.exp(x - jnp.max(x, axis=1, keepdims=True))
            return exp_x / jnp.sum(exp_x, axis=1, keepdims=True)

        means, stds, skews, kurts = compute_stats(mat_jax)
        probs = softmax_rows(mat_jax)

        return {
            "jax_row_mean": np.array(means).tolist(),
            "jax_row_std": np.array(stds).tolist(),
            "jax_skewness": np.array(skews).tolist(),
            "jax_kurtosis": np.array(kurts).tolist(),
        }

    def _numpy_compute(self, mat: np.ndarray, col_names: list[str]) -> dict:
        """NumPy fallback hesaplamalar."""
        row_mean = np.mean(mat, axis=1)
        row_std = np.std(mat, axis=1)

        centered = (mat.T - row_mean).T / (row_std[:, None] + 1e-8)
        skew = np.mean(centered ** 3, axis=1)
        kurt = np.mean(centered ** 4, axis=1) - 3

        return {
            "jax_row_mean": row_mean.tolist(),
            "jax_row_std": row_std.tolist(),
            "jax_skewness": skew.tolist(),
            "jax_kurtosis": kurt.tolist(),
        }

    def batch_probability(self, logits: np.ndarray) -> np.ndarray:
        """Logit dizisini olasılığa dönüştürür (batch)."""
        if JAX_AVAILABLE:
            @jit
            def softmax(x):
                exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
                return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)
            return np.array(softmax(jnp.array(logits)))
        else:
            exp_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
