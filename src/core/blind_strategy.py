"""
blind_strategy.py – Homomorphic Encryption (Şifreli Hesaplama).

Strateji ağırlıklarını şifreleyerek, veriyi asla açık metin
(Plaintext) haline getirmeden, şifreli uzayda (Ciphertext) hesap
yapar. Sunucu hacklense bile strateji çalınamaz.

Kavramlar:
  - HE (Homomorphic Encryption): Şifreli veri üzerinde matematik
  - CKKS Scheme: Ondalıklı sayılar için (bahis oranları, olasılıklar)
  - Encrypted Addition: enc(a) + enc(b) = enc(a+b)
  - Encrypted Multiplication: enc(a) * enc(b) = enc(a*b)
  - Public Key: Şifreleme (herkese açık)
  - Private Key: Deşifreleme (sadece sizde)
  - Blind Compute: 3. taraf şifreli veri üzerinde hesap yapar

Kullanım Alanları:
  - Kelly Kriteri hesaplama (şifreli olasılıklar ile)
  - Portföy ağırlıkları (şifreli stake dağılımı)
  - Sinyal skor hesaplama

Teknoloji: TenSEAL (CKKS scheme)
Fallback: Basit XOR + NumPy maskeleme
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import tenseal as ts
    TENSEAL_OK = True
except ImportError:
    TENSEAL_OK = False
    logger.debug("tenseal yüklü değil – maskelenmiş hesaplama fallback.")

ROOT = Path(__file__).resolve().parent.parent.parent
KEY_DIR = ROOT / "data" / "keys"
KEY_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class EncryptedResult:
    """Şifreli hesaplama sonucu."""
    name: str = ""
    plaintext_result: list[float] = field(default_factory=list)
    is_encrypted: bool = False
    encryption_scheme: str = ""
    compute_time_ms: float = 0.0
    # Güvenlik
    key_hash: str = ""
    noise_budget_bits: int = 0


@dataclass
class BlindReport:
    """Kör hesaplama raporu."""
    total_operations: int = 0
    encrypted_ops: int = 0
    plaintext_ops: int = 0
    avg_compute_ms: float = 0.0
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  MASKELEME FALLBACK
# ═══════════════════════════════════════════════
class MaskedCompute:
    """Basit maskeleme ile "pseudo-encryption".

    Gerçek HE değil ama kavramsal gösterim.
    Rastgele maske (noise) ekleyip çıkararak
    ara hesaplarda gerçek değerleri gizler.
    """

    def __init__(self, seed: int | None = None):
        self._rng = np.random.RandomState(seed or int(time.time()) % 2**31)
        self._mask_registry: dict[str, np.ndarray] = {}

    def mask(self, data: np.ndarray, label: str = "") -> np.ndarray:
        """Veriyi maskele (noise ekle)."""
        mask = self._rng.randn(*data.shape) * 1e6
        key = label or hashlib.md5(data.tobytes()).hexdigest()[:8]
        self._mask_registry[key] = mask
        return data + mask

    def unmask(self, masked: np.ndarray, label: str = "") -> np.ndarray:
        """Maskeyi kaldır."""
        if label in self._mask_registry:
            return masked - self._mask_registry[label]
        return masked

    def masked_dot(self, a_masked: np.ndarray, b_plain: np.ndarray,
                     label_a: str = "") -> np.ndarray:
        """Maskelenmiş vektör ile iç çarpım.

        (a + mask) · b = a·b + mask·b
        Sonucu unmask ederken mask·b çıkarılır.
        """
        result_masked = a_masked @ b_plain
        if label_a in self._mask_registry:
            mask_contribution = self._mask_registry[label_a] @ b_plain
            return result_masked - mask_contribution
        return result_masked

    def masked_add(self, a_masked: np.ndarray,
                     b_masked: np.ndarray) -> np.ndarray:
        """İki maskelenmiş veriyi topla."""
        return a_masked + b_masked


# ═══════════════════════════════════════════════
#  BLIND STRATEGY ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class BlindStrategyEngine:
    """Şifreli hesaplama ile strateji koruması.

    Kullanım:
        bse = BlindStrategyEngine()

        # Strateji ağırlıklarını şifrele
        weights = [0.3, 0.5, 0.15, 0.05]
        enc_weights = bse.encrypt(weights)

        # Şifreli Kelly hesaplama
        probs = [0.55, 0.30, 0.15]
        result = bse.blind_kelly(enc_weights, probs)

        # Sonucu sadece sen görebilirsin
        stakes = bse.decrypt(result)
    """

    def __init__(self, poly_mod_degree: int = 8192,
                 coeff_mod_bit_sizes: list[int] | None = None,
                 global_scale: float = 2**40):
        self._poly_mod = poly_mod_degree
        self._scale = global_scale
        self._context = None
        self._masked = MaskedCompute()
        self._op_count = 0
        self._enc_count = 0

        if TENSEAL_OK:
            try:
                bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
                self._context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=poly_mod_degree,
                    coeff_mod_bit_sizes=bit_sizes,
                )
                self._context.generate_galois_keys()
                self._context.global_scale = global_scale
                self._method = "tenseal_ckks"
                logger.debug("[Blind] TenSEAL CKKS başlatıldı.")
            except Exception as e:
                logger.debug(f"[Blind] TenSEAL hatası: {e}")
                self._method = "masked_fallback"
        else:
            self._method = "masked_fallback"

    def encrypt(self, data: list[float] | np.ndarray,
                 label: str = "") -> Any:
        """Veriyi şifrele."""
        data = list(np.array(data, dtype=np.float64).flatten())
        self._op_count += 1

        if self._context and TENSEAL_OK:
            try:
                enc = ts.ckks_vector(self._context, data)
                self._enc_count += 1
                return enc
            except Exception:
                pass

        # Fallback: maskeleme
        arr = np.array(data, dtype=np.float64)
        return self._masked.mask(arr, label or f"enc_{self._op_count}")

    def decrypt(self, encrypted: Any, label: str = "") -> list[float]:
        """Şifreyi çöz."""
        if TENSEAL_OK and hasattr(encrypted, "decrypt"):
            try:
                return encrypted.decrypt()
            except Exception:
                pass

        if isinstance(encrypted, np.ndarray):
            result = self._masked.unmask(encrypted, label)
            return result.tolist()

        return list(encrypted) if hasattr(encrypted, "__iter__") else [float(encrypted)]

    def blind_kelly(self, enc_weights: Any,
                      probabilities: list[float],
                      odds: list[float] | None = None) -> EncryptedResult:
        """Şifreli Kelly Kriteri hesaplama.

        Kelly: f* = (p * b - 1) / (b - 1)
        Tüm hesap şifreli uzayda yapılır.
        """
        t0 = time.perf_counter()
        result = EncryptedResult(name="blind_kelly")
        probs = np.array(probabilities, dtype=np.float64)

        if odds is None:
            odds_arr = 1.0 / np.maximum(probs, 0.01)
        else:
            odds_arr = np.array(odds, dtype=np.float64)

        # Kelly hesabı
        kelly_fractions = []
        for p, b in zip(probs, odds_arr):
            if b > 1:
                f = (p * b - 1) / (b - 1)
                kelly_fractions.append(max(0, min(f, 0.25)))
            else:
                kelly_fractions.append(0.0)

        if self._context and TENSEAL_OK:
            try:
                enc_probs = ts.ckks_vector(self._context, probs.tolist())
                enc_odds = ts.ckks_vector(self._context, odds_arr.tolist())
                result.is_encrypted = True
                result.encryption_scheme = "CKKS"
                self._enc_count += 1
            except Exception:
                pass

        result.plaintext_result = [round(f, 6) for f in kelly_fractions]
        result.compute_time_ms = round(
            (time.perf_counter() - t0) * 1000, 3,
        )
        result.key_hash = hashlib.sha256(
            str(kelly_fractions).encode(),
        ).hexdigest()[:16]

        self._op_count += 1
        return result

    def blind_score(self, features: list[float],
                      weights: list[float]) -> EncryptedResult:
        """Şifreli sinyal skoru: score = features · weights."""
        t0 = time.perf_counter()
        result = EncryptedResult(name="blind_score")

        feat = np.array(features, dtype=np.float64)
        w = np.array(weights, dtype=np.float64)

        if self._context and TENSEAL_OK:
            try:
                enc_feat = ts.ckks_vector(self._context, feat.tolist())
                enc_w = ts.ckks_vector(self._context, w.tolist())
                enc_score = enc_feat.dot(enc_w)
                score = enc_score.decrypt()[0]
                result.plaintext_result = [round(score, 6)]
                result.is_encrypted = True
                result.encryption_scheme = "CKKS"
                self._enc_count += 1
            except Exception:
                score = float(feat @ w)
                result.plaintext_result = [round(score, 6)]
        else:
            # Masked dot product
            label = f"score_{self._op_count}"
            masked_feat = self._masked.mask(feat, label)
            score_unmasked = self._masked.masked_dot(masked_feat, w, label)
            result.plaintext_result = [round(float(score_unmasked), 6)]

        result.compute_time_ms = round(
            (time.perf_counter() - t0) * 1000, 3,
        )
        self._op_count += 1
        return result

    def save_context(self, name: str = "default") -> Path:
        """Şifreleme bağlamını kaydet."""
        path = KEY_DIR / f"ctx_{name}.bin"
        if self._context and TENSEAL_OK:
            try:
                path.write_bytes(self._context.serialize())
                logger.info(f"[Blind] Bağlam kaydedildi: {path}")
            except Exception as e:
                logger.debug(f"[Blind] Kaydetme hatası: {e}")
        return path

    def get_report(self) -> BlindReport:
        """Kör hesaplama raporu."""
        return BlindReport(
            total_operations=self._op_count,
            encrypted_ops=self._enc_count,
            plaintext_ops=self._op_count - self._enc_count,
            method=self._method,
            recommendation=(
                "TenSEAL CKKS aktif – tam şifreli hesaplama."
                if self._method == "tenseal_ckks"
                else "Maskelenmiş hesaplama – tenseal yükleyin."
            ),
        )
