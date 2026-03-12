"""
xai_explainer.py – SHAP ile Açıklanabilir Yapay Zeka (XAI).

"Neden?" bilmeden para yatırılmaz.
Model %65 Ev Sahibi diyor – ama sebep ne?

SHAP (SHapley Additive exPlanations):
  - LightGBM ve diğer modellerin feature importance'ını hesaplar
  - Her maç için bir "Decision Plot" / "Waterfall Plot" oluşturur
  - Metin açıklaması: "Model %65 Ev diyor çünkü:
    1. Deplasman xG düşük (+%15)
    2. Ev sahibi son 5 maç yenilmedi (+%10)"
  - Görseli Telegram'a gönderir

Kara Kutu → Şeffaf Kutu
"""
from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    logger.warning("shap yüklü değil – XAI sadeleştirilmiş modda.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

# Türkçe feature isim haritası
FEATURE_TR = {
    "home_impl_prob": "Ev sahibi oran olasılığı",
    "draw_impl_prob": "Beraberlik oran olasılığı",
    "away_impl_prob": "Deplasman oran olasılığı",
    "margin": "Bahisçi marjı",
    "xg_diff": "xG farkı (Ev-Dep)",
    "xg_total": "Toplam xG",
    "def_xg_diff": "Defansif xG farkı",
    "home_win_rate": "Ev galibiyet oranı",
    "away_win_rate": "Dep. galibiyet oranı",
    "form_diff": "Form puan farkı",
    "goal_diff": "Gol farkı (son maçlar)",
    "possession_diff": "Topla oynama farkı",
    "rank_diff": "Sıralama farkı",
    "home_form_pts": "Ev formu (puan)",
    "away_form_pts": "Dep. formu (puan)",
    "momentum_home": "Ev sahibi momentum",
    "momentum_away": "Deplasman momentum",
    "entropy": "Oran entropisi",
    "home_xg": "Ev sahibi xG",
    "away_xg": "Deplasman xG",
    "home_goals_avg": "Ev gol ortalaması",
    "away_goals_avg": "Dep. gol ortalaması",
}


class XAIExplainer:
    """SHAP tabanlı model açıklayıcı.

    Kullanım:
        explainer = XAIExplainer()
        explainer.set_model(lightgbm_model, X_train)
        # Tek maç açıklaması
        result = explainer.explain(match_features)
        # Telegram'a görsel gönder
        fig = explainer.plot_waterfall(match_features)
        await chart_sender.send_chart(fig, "GS vs FB Karar Analizi")
    """

    def __init__(self, max_display: int = 10):
        self._model = None
        self._explainer = None
        self._feature_names: list[str] = []
        self._max_display = max_display
        self._background_data = None
        logger.debug(f"XAIExplainer başlatıldı (SHAP={'✓' if SHAP_OK else '✗'}).")

    def set_model(self, model: Any, X_train: np.ndarray | None = None,
                  feature_names: list[str] | None = None):
        """Model ve arka plan verisini ayarla."""
        self._model = model
        self._feature_names = feature_names or []

        if SHAP_OK and model is not None:
            try:
                # LightGBM / XGBoost → TreeExplainer (hızlı)
                self._explainer = shap.TreeExplainer(model)
                logger.info("[XAI] TreeExplainer aktif.")
            except Exception as e:
                logger.debug(f"Exception caught: {e}")
                try:
                    # Genel model → KernelExplainer (yavaş ama evrensel)
                    if X_train is not None:
                        background = shap.sample(X_train, 100)
                    else:
                        background = X_train
                    self._explainer = shap.KernelExplainer(
                        model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                        background,
                    )
                    logger.info("[XAI] KernelExplainer aktif.")
                except Exception as e:
                    logger.warning(f"[XAI] Explainer oluşturulamadı: {e}")

    def explain(self, features: np.ndarray | dict,
                class_idx: int = 0) -> dict:
        """Tek bir maç tahminini açıkla.

        Returns:
            {
                "explanation_text": str,      # İnsan okunur metin
                "top_features": [...],        # En etkili feature'lar
                "shap_values": [...],         # Ham SHAP değerleri
                "base_value": float,          # Baz değer
                "prediction": float,          # Model tahmini
            }
        """
        if isinstance(features, dict):
            if not self._feature_names:
                self._feature_names = list(features.keys())
            features = np.array([[features.get(k, 0) for k in self._feature_names]])

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._explainer and SHAP_OK:
            return self._explain_shap(features, class_idx)
        else:
            return self._explain_heuristic(features)

    def _explain_shap(self, X: np.ndarray, class_idx: int) -> dict:
        """SHAP ile açıklama üret."""
        try:
            shap_values = self._explainer.shap_values(X)

            # Multi-class: ilgili sınıfın değerlerini al
            if isinstance(shap_values, list):
                sv = shap_values[class_idx][0]
            else:
                sv = shap_values[0]

            base_value = (
                self._explainer.expected_value[class_idx]
                if isinstance(self._explainer.expected_value, (list, np.ndarray))
                else self._explainer.expected_value
            )

            # Feature isim - değer - etki eşlemesi
            top_features = self._rank_features(sv, X[0])
            text = self._generate_text(top_features, base_value)

            return {
                "explanation_text": text,
                "top_features": top_features,
                "shap_values": sv.tolist() if hasattr(sv, "tolist") else list(sv),
                "base_value": float(base_value),
                "method": "shap",
            }
        except Exception as e:
            logger.warning(f"[XAI] SHAP hesaplama hatası: {e}")
            return self._explain_heuristic(X)

    def _explain_heuristic(self, X: np.ndarray) -> dict:
        """SHAP yokken basit feature importance tahmini."""
        values = X[0]
        feature_impacts = []

        for i, val in enumerate(values):
            name = self._feature_names[i] if i < len(self._feature_names) else f"F{i}"
            impact = float(abs(val - 0.5) * 0.1) if val != 0 else 0
            feature_impacts.append({
                "name": name,
                "name_tr": FEATURE_TR.get(name, name),
                "value": float(val),
                "impact": impact,
                "direction": "positive" if val > 0.5 else "negative",
            })

        feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top = feature_impacts[:self._max_display]
        text = self._generate_text(top, 0.33)

        return {
            "explanation_text": text,
            "top_features": top,
            "shap_values": [],
            "base_value": 0.33,
            "method": "heuristic",
        }

    def _rank_features(self, shap_values: np.ndarray,
                        feature_values: np.ndarray) -> list[dict]:
        """Feature'ları etkiye göre sırala."""
        impacts = []
        for i, (sv, fv) in enumerate(zip(shap_values, feature_values)):
            name = self._feature_names[i] if i < len(self._feature_names) else f"F{i}"
            impacts.append({
                "name": name,
                "name_tr": FEATURE_TR.get(name, name),
                "value": float(fv),
                "impact": float(sv),
                "impact_pct": float(abs(sv) * 100),
                "direction": "positive" if sv > 0 else "negative",
            })

        impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return impacts[:self._max_display]

    def _generate_text(self, top_features: list[dict],
                        base_value: float) -> str:
        """İnsan okunur açıklama metni oluştur."""
        lines = []
        for i, f in enumerate(top_features[:5], 1):
            name_tr = f.get("name_tr", f["name"])
            impact_pct = f.get("impact_pct", abs(f.get("impact", 0)) * 100)
            direction = "↑" if f["direction"] == "positive" else "↓"
            lines.append(f"{i}. {name_tr}: {direction} {impact_pct:.1f}%")

        return "\n".join(lines) if lines else "Açıklama üretilemedi."

    # ═══════════════════════════════════════════
    #  GÖRSELLEŞTİRME
    # ═══════════════════════════════════════════
    def plot_waterfall(self, features: np.ndarray | dict,
                       class_idx: int = 0,
                       title: str = "Karar Analizi") -> Any:
        """SHAP Waterfall Plot oluştur."""
        if not (SHAP_OK and MPL_OK and self._explainer):
            return self._plot_bar_fallback(features, title)

        if isinstance(features, dict):
            X = np.array([[features.get(k, 0) for k in self._feature_names]])
        else:
            X = features if features.ndim == 2 else features.reshape(1, -1)

        try:
            explanation = self._explainer(X)
            if isinstance(explanation, list):
                explanation = explanation[class_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            plt.style.use("dark_background")
            shap.plots.waterfall(explanation[0], max_display=self._max_display,
                                  show=False)
            fig = plt.gcf()
            fig.suptitle(title, fontsize=14, fontweight="bold", color="white")
            fig.set_facecolor("#1a1a2e")
            return fig
        except Exception as e:
            logger.debug(f"[XAI] Waterfall plot hatası: {e}")
            return self._plot_bar_fallback(features, title)

    def plot_force(self, features: np.ndarray | dict,
                   class_idx: int = 0) -> Any:
        """SHAP Force Plot oluştur (HTML)."""
        if not (SHAP_OK and self._explainer):
            return None

        if isinstance(features, dict):
            X = np.array([[features.get(k, 0) for k in self._feature_names]])
        else:
            X = features if features.ndim == 2 else features.reshape(1, -1)

        try:
            shap_values = self._explainer.shap_values(X)
            if isinstance(shap_values, list):
                sv = shap_values[class_idx]
            else:
                sv = shap_values

            base = (
                self._explainer.expected_value[class_idx]
                if isinstance(self._explainer.expected_value, (list, np.ndarray))
                else self._explainer.expected_value
            )

            return shap.force_plot(
                base, sv[0], X[0],
                feature_names=self._feature_names,
                matplotlib=True, show=False,
            )
        except Exception as e:
            logger.debug(f"Exception caught: {e}")
            return None

    def _plot_bar_fallback(self, features: Any, title: str) -> Any:
        """SHAP yokken basit bar chart."""
        if not MPL_OK:
            return None

        result = self.explain(features)
        top = result.get("top_features", [])
        if not top:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        names = [f.get("name_tr", f["name"])[:25] for f in top]
        impacts = [f.get("impact_pct", abs(f.get("impact", 0)) * 100) for f in top]
        colors = ["#e94560" if f["direction"] == "negative" else "#0f3460" for f in top]

        y_pos = range(len(names))
        ax.barh(y_pos, impacts, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, color="white", fontsize=9)
        ax.set_xlabel("Etki (%)", color="white")
        ax.set_title(title, color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_color("#333")

        plt.tight_layout()
        return fig

    # ═══════════════════════════════════════════
    #  TELEGRAM ENTEGRASYONU
    # ═══════════════════════════════════════════
    async def explain_and_send(self, features: np.ndarray | dict,
                                match_title: str,
                                chart_sender=None,
                                class_idx: int = 0) -> dict:
        """Açıklama üret + Waterfall plot → Telegram'a gönder."""
        result = self.explain(features, class_idx)

        text = (
            f"🔍 <b>{match_title} – XAI Analiz</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"<b>Model neden bu kararı verdi?</b>\n\n"
            f"<code>{result['explanation_text']}</code>\n\n"
            f"<i>Metod: {result.get('method', 'unknown')}</i>"
        )

        if chart_sender:
            fig = self.plot_waterfall(features, class_idx, match_title)
            if fig:
                await chart_sender.send_chart(fig, f"🔍 {match_title}")

        return {**result, "telegram_text": text}
