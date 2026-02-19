"""
prophet_seasonality.py – Mevsimsellik Analizi (Time Series Decomposition).

Futbol takımları kışın farklı, baharda farklı oynar.
Lig sonu yorgunluğu veya "Noel Tatili" dönüşü performans düşüşleri
birer "Mevsimsellik" (Seasonality) etkisidir.

Prophet kullanarak:
  - Trend      → Uzun vadeli yükseliş / düşüş
  - Yearly     → Yıl içi mevsimsellik (Şubat çukuru, Mayıs zirvesi)
  - Weekly     → Hafta sonu / hafta içi etkisi
  - Holidays   → Lig arası, milli maç molaları
  - Residual   → Açıklanamayan gürültü

Sinyal:
  "Bu takım Şubat aylarında tarihsel olarak düşüş yaşıyor mu?"
  Eğer evet → favori olsa bile "AVOID" sinyali üret.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False
    logger.info("prophet yüklü değil – heuristic mevsimsellik aktif.")


@dataclass
class SeasonalityResult:
    """Mevsimsellik analiz sonucu."""
    team: str
    current_date: str
    trend: float = 0.0           # Uzun vadeli trend (-1 to 1)
    seasonal_effect: float = 0.0  # Mevsimsel düzeltme (-0.5 to 0.5)
    is_negative_season: bool = False  # Bu dönemde düşüş var mı?
    avoid_signal: bool = False    # Favori olsa bile bahisten kaçın
    confidence: float = 0.0
    decomposition: dict = field(default_factory=dict)
    method: str = "heuristic"
    explanation: str = ""


@dataclass
class TurkishFootballCalendar:
    """Türk futbol takvimi mevsimsellik bilgisi.

    Ampirik gözlemler:
      - Ocak: Kış transfer dönemi, yeni oyuncu uyumu → Volatilite yüksek
      - Şubat: Soğuk hava, saha kalitesi düşük → Gol sayıları düşer
      - Mart-Nisan: Şampiyonluk yarışı kızışır → Favori performansı artar
      - Mayıs: Son haftalar, küme düşme korkusu → Underdog sürprizleri
      - Haziran-Temmuz: Lig arası
      - Ağustos: Sezon başı, formlar belli değil → Volatilite yüksek
      - Eylül-Ekim: Takımlar oturur → Model güvenilirliği artar
      - Kasım: Milli maç molaları → Sakatlık riski artar
      - Aralık: Yoğun fikstür → Rotasyon fazla → Sürpriz artar
    """

    MONTH_EFFECTS: dict[int, dict] = field(default_factory=lambda: {
        1:  {"effect": -0.05, "volatility": 1.3, "label": "Kış transferi uyum dönemi"},
        2:  {"effect": -0.10, "volatility": 1.1, "label": "Soğuk hava düşüşü"},
        3:  {"effect":  0.05, "volatility": 0.9, "label": "Şampiyonluk ivmesi"},
        4:  {"effect":  0.08, "volatility": 0.8, "label": "Kritik dönem – yüksek motivasyon"},
        5:  {"effect": -0.03, "volatility": 1.4, "label": "Son haftalar – sürpriz riski"},
        6:  {"effect":  0.00, "volatility": 0.5, "label": "Lig arası"},
        7:  {"effect":  0.00, "volatility": 0.5, "label": "Lig arası"},
        8:  {"effect": -0.08, "volatility": 1.5, "label": "Sezon başı belirsizliği"},
        9:  {"effect":  0.03, "volatility": 1.0, "label": "Takımlar oturmaya başlıyor"},
        10: {"effect":  0.05, "volatility": 0.9, "label": "Stabil dönem"},
        11: {"effect": -0.02, "volatility": 1.1, "label": "Milli maç molası riski"},
        12: {"effect": -0.06, "volatility": 1.2, "label": "Yoğun fikstür rotasyonu"},
    })


class ProphetSeasonalityAnalyzer:
    """Prophet ile mevsimsellik analizi.

    Kullanım:
        analyzer = ProphetSeasonalityAnalyzer()
        # 3 yıllık performans verisi yükle
        analyzer.fit("Galatasaray", historical_data)
        # Bugünkü mevsimsellik etkisini sorgula
        result = analyzer.predict("Galatasaray")
    """

    # Mevsimsellik eşikleri
    AVOID_THRESHOLD = -0.12       # Bu değerin altında → AVOID sinyali
    NEGATIVE_THRESHOLD = -0.05    # Negatif sezonsal etki eşiği
    MIN_DATAPOINTS = 30           # Minimum veri noktası (Prophet için)
    FORECAST_HORIZON = 14         # Gün cinsinden tahmin ufku

    def __init__(self):
        self._models: dict[str, Any] = {}   # team → fitted Prophet model
        self._data: dict[str, Any] = {}     # team → training DataFrame
        self._calendar = TurkishFootballCalendar()
        self._cache: dict[str, SeasonalityResult] = {}
        logger.debug("ProphetSeasonalityAnalyzer başlatıldı.")

    # ═══════════════════════════════════════════
    #  VERİ YÜKLEME & MODEL EĞİTME
    # ═══════════════════════════════════════════
    def fit(self, team: str, data: list[dict] | Any,
            date_col: str = "date", value_col: str = "points") -> bool:
        """Takımın tarihsel performans verisini yükle ve Prophet eğit.

        data: [
            {"date": "2023-09-15", "points": 3},
            {"date": "2023-09-22", "points": 1},
            ...
        ]
        veya Pandas DataFrame.
        """
        if not PANDAS_OK:
            logger.warning("[Prophet] pandas yüklü değil.")
            return False

        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif hasattr(data, "to_pandas"):
                df = data.to_pandas()
            else:
                df = pd.DataFrame(data)

            if date_col not in df.columns or value_col not in df.columns:
                logger.warning(f"[Prophet] {team}: '{date_col}' veya '{value_col}' sütunu yok.")
                return False

            # Prophet formatı: ds (tarih), y (değer)
            prophet_df = pd.DataFrame({
                "ds": pd.to_datetime(df[date_col]),
                "y": pd.to_numeric(df[value_col], errors="coerce"),
            }).dropna().sort_values("ds").reset_index(drop=True)

            if len(prophet_df) < self.MIN_DATAPOINTS:
                logger.info(
                    f"[Prophet] {team}: {len(prophet_df)} veri noktası < {self.MIN_DATAPOINTS}. "
                    f"Heuristic mod kullanılacak."
                )
                self._data[team] = prophet_df
                return False

            self._data[team] = prophet_df

            if not PROPHET_OK:
                logger.info(f"[Prophet] {team}: prophet yüklü değil – heuristic mod.")
                return False

            # Prophet model eğit
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                seasonality_mode="additive",
            )

            # Türk ligi özel mevsimsellik (Süper Lig takvimi)
            model.add_seasonality(
                name="half_season",
                period=365.25 / 2,
                fourier_order=5,
            )

            # Lig araları (tatil olarak ekle)
            holidays = self._get_league_breaks()
            if holidays is not None and len(holidays) > 0:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10,
                    seasonality_mode="additive",
                    holidays=holidays,
                )
                model.add_seasonality(
                    name="half_season",
                    period=365.25 / 2,
                    fourier_order=5,
                )

            model.fit(prophet_df)
            self._models[team] = model

            logger.success(
                f"[Prophet] {team}: {len(prophet_df)} veri noktası ile eğitildi."
            )
            return True

        except Exception as e:
            logger.warning(f"[Prophet] {team} eğitim hatası: {e}")
            return False

    # ═══════════════════════════════════════════
    #  TAHMİN VE MEVSİMSELLİK ÇIKARIMI
    # ═══════════════════════════════════════════
    def predict(self, team: str,
                target_date: str | datetime | None = None) -> SeasonalityResult:
        """Belirli bir tarih için mevsimsellik etkisini hesapla."""
        if target_date is None:
            target_date = datetime.now()
        elif isinstance(target_date, str):
            target_date = datetime.fromisoformat(target_date)

        cache_key = f"{team}_{target_date.strftime('%Y-%m-%d')}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        model = self._models.get(team)
        if model and PROPHET_OK and PANDAS_OK:
            result = self._predict_prophet(team, model, target_date)
        else:
            result = self._predict_heuristic(team, target_date)

        self._cache[cache_key] = result
        return result

    def _predict_prophet(self, team: str, model: Any,
                          target_date: datetime) -> SeasonalityResult:
        """Prophet model ile tahmin."""
        try:
            future = model.make_future_dataframe(
                periods=self.FORECAST_HORIZON, freq="D",
            )

            forecast = model.predict(future)

            # Hedef tarihe en yakın satırı bul
            target_str = target_date.strftime("%Y-%m-%d")
            forecast["ds_str"] = forecast["ds"].dt.strftime("%Y-%m-%d")
            row = forecast[forecast["ds_str"] == target_str]

            if row.empty:
                row = forecast.iloc[-1:]

            trend = float(row["trend"].values[0])
            yearly = float(row.get("yearly", pd.Series([0])).values[0])
            weekly = float(row.get("weekly", pd.Series([0])).values[0])
            yhat = float(row["yhat"].values[0])

            # Normalize
            data_mean = self._data.get(team, pd.DataFrame({"y": [1]}))["y"].mean() or 1
            seasonal_effect = (yearly + weekly) / data_mean
            trend_normalized = (trend - data_mean) / data_mean

            is_negative = seasonal_effect < self.NEGATIVE_THRESHOLD
            avoid = seasonal_effect < self.AVOID_THRESHOLD

            explanation = self._generate_explanation(
                team, target_date, seasonal_effect, trend_normalized, "prophet",
            )

            return SeasonalityResult(
                team=team,
                current_date=target_str,
                trend=round(trend_normalized, 4),
                seasonal_effect=round(seasonal_effect, 4),
                is_negative_season=is_negative,
                avoid_signal=avoid,
                confidence=0.75,
                decomposition={
                    "trend": trend,
                    "yearly": yearly,
                    "weekly": weekly,
                    "yhat": yhat,
                },
                method="prophet",
                explanation=explanation,
            )

        except Exception as e:
            logger.warning(f"[Prophet] {team} tahmin hatası: {e}")
            return self._predict_heuristic(team, target_date)

    def _predict_heuristic(self, team: str,
                            target_date: datetime) -> SeasonalityResult:
        """Prophet yoksa → Türk ligi takvimi ile heuristic tahmin."""
        month_names = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
            5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
            9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
        }
        month = target_date.month
        month_data = self._calendar.MONTH_EFFECTS.get(month, {})
        label = month_data.get("label", "")

        # ── FIX: Hiç tarihsel veri yoksa → generic takvim etkisi UYGULAMA ──
        # Generic -10% gibi değerler gerçek olmayan sinyale yol açıyor.
        team_data = self._data.get(team)
        if team_data is None:
            return SeasonalityResult(
                team=team,
                current_date=target_date.strftime("%Y-%m-%d"),
                trend=0.0,
                seasonal_effect=0.0,
                is_negative_season=False,
                avoid_signal=False,
                confidence=0.10,
                decomposition={
                    "method": "no_data",
                    "month": month,
                    "month_label": label,
                },
                method="neutral_fallback",
                explanation=(
                    f"⚪ {team} – {month_names.get(month, '?')}: "
                    f"Tarihsel veri yok, nötr varsayım uygulandı."
                ),
            )

        effect = month_data.get("effect", 0.0)
        volatility = month_data.get("volatility", 1.0)

        # Eğer tarihi veri varsa, takıma özel ay ortalaması hesapla
        if team_data is not None and PANDAS_OK and len(team_data) > 10:
            try:
                team_data = team_data.copy()
                team_data["month"] = team_data["ds"].dt.month
                month_avg = team_data[team_data["month"] == month]["y"].mean()
                overall_avg = team_data["y"].mean()

                if overall_avg > 0 and not np.isnan(month_avg):
                    team_effect = (month_avg - overall_avg) / overall_avg
                    effect = team_effect * 0.7 + effect * 0.3
            except Exception:
                pass

        is_negative = effect < self.NEGATIVE_THRESHOLD
        avoid = effect < self.AVOID_THRESHOLD

        explanation = self._generate_explanation(
            team, target_date, effect, 0.0, "heuristic",
        )

        return SeasonalityResult(
            team=team,
            current_date=target_date.strftime("%Y-%m-%d"),
            trend=0.0,
            seasonal_effect=round(effect, 4),
            is_negative_season=is_negative,
            avoid_signal=avoid,
            confidence=0.40 if team_data is not None else 0.25,
            decomposition={
                "month": month,
                "month_label": label,
                "volatility": volatility,
            },
            method="heuristic",
            explanation=explanation,
        )

    # ═══════════════════════════════════════════
    #  MAÇ ANALİZİ ENTEGRASYONU
    # ═══════════════════════════════════════════
    def analyze_match(self, home_team: str, away_team: str,
                       match_date: str | datetime | None = None) -> dict:
        """İki takım için mevsimsellik karşılaştırması.

        Returns:
            {
                "home_seasonal": SeasonalityResult,
                "away_seasonal": SeasonalityResult,
                "home_adjustment": float,  # xG çarpanı
                "away_adjustment": float,
                "avoid_match": bool,       # Her iki takım da negatif mi
                "insight": str,
            }
        """
        home_result = self.predict(home_team, match_date)
        away_result = self.predict(away_team, match_date)

        # xG adjustment: mevsimsel etki → çarpan
        home_adj = 1.0 + home_result.seasonal_effect
        away_adj = 1.0 + away_result.seasonal_effect

        # Clamp
        home_adj = max(0.7, min(home_adj, 1.3))
        away_adj = max(0.7, min(away_adj, 1.3))

        avoid = home_result.avoid_signal and away_result.avoid_signal

        # İçgörü üret
        insights = []
        if home_result.is_negative_season:
            insights.append(
                f"⚠️ {home_team} bu dönemde tarihsel olarak düşüş yaşıyor "
                f"(etki: {home_result.seasonal_effect:+.1%})."
            )
        if away_result.is_negative_season:
            insights.append(
                f"⚠️ {away_team} bu dönemde tarihsel olarak düşüş yaşıyor "
                f"(etki: {away_result.seasonal_effect:+.1%})."
            )
        if home_result.seasonal_effect > 0.05:
            insights.append(
                f"📈 {home_team} bu dönemde tarihsel olarak güçlü "
                f"(etki: {home_result.seasonal_effect:+.1%})."
            )
        if away_result.seasonal_effect > 0.05:
            insights.append(
                f"📈 {away_team} bu dönemde tarihsel olarak güçlü "
                f"(etki: {away_result.seasonal_effect:+.1%})."
            )
        if avoid:
            insights.append("🚫 Her iki takım da negatif sezonda – AVOID önerisi!")

        return {
            "home_seasonal": home_result,
            "away_seasonal": away_result,
            "home_adjustment": round(home_adj, 3),
            "away_adjustment": round(away_adj, 3),
            "avoid_match": avoid,
            "insight": " | ".join(insights) if insights else "Mevsimsel anomali yok.",
        }

    # ═══════════════════════════════════════════
    #  TOPLU TAHMİN (Tüm takımlar)
    # ═══════════════════════════════════════════
    async def predict_all_teams(self, target_date: str | datetime | None = None
                                 ) -> dict[str, SeasonalityResult]:
        """Yüklenmiş tüm takımlar için mevsimsellik tahmini."""
        results = {}
        for team in list(self._data.keys()):
            results[team] = self.predict(team, target_date)
        return results

    # ═══════════════════════════════════════════
    #  YARDIMCI
    # ═══════════════════════════════════════════
    @staticmethod
    def _get_league_breaks() -> Any:
        """Türk Süper Lig ara dönemleri (kış + yaz)."""
        if not PANDAS_OK:
            return None

        holidays = []
        for year in range(2020, 2027):
            # Kış arası (yaklaşık 3 hafta)
            holidays.append({
                "holiday": "winter_break",
                "ds": f"{year}-01-10",
                "lower_window": -7,
                "upper_window": 14,
            })
            # Yaz arası (yaklaşık 2.5 ay)
            holidays.append({
                "holiday": "summer_break",
                "ds": f"{year}-06-15",
                "lower_window": -7,
                "upper_window": 60,
            })
            # Milli maç molaları (eylül, ekim, kasım)
            for month in (9, 10, 11):
                holidays.append({
                    "holiday": "international_break",
                    "ds": f"{year}-{month:02d}-05",
                    "lower_window": -2,
                    "upper_window": 7,
                })

        return pd.DataFrame(holidays)

    @staticmethod
    def _generate_explanation(team: str, date: datetime,
                               seasonal_effect: float,
                               trend: float,
                               method: str) -> str:
        """İnsan-okunabilir Türkçe açıklama."""
        month_names = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
            5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
            9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
        }
        month = month_names.get(date.month, "?")

        if seasonal_effect < -0.12:
            quality = "GÜÇLÜ NEGATİF dönemde"
            emoji = "🔴"
        elif seasonal_effect < -0.05:
            quality = "negatif dönemde"
            emoji = "🟠"
        elif seasonal_effect > 0.08:
            quality = "GÜÇLÜ POZİTİF dönemde"
            emoji = "🟢"
        elif seasonal_effect > 0.03:
            quality = "pozitif dönemde"
            emoji = "🔵"
        else:
            quality = "nötr dönemde"
            emoji = "⚪"

        return (
            f"{emoji} {team} – {month} ayı analizi: {quality}. "
            f"Mevsimsel etki: {seasonal_effect:+.1%}. "
            f"Metod: {method}."
        )

    def clear_cache(self):
        self._cache.clear()

    @property
    def fitted_teams(self) -> list[str]:
        """Prophet ile eğitilmiş takımlar."""
        return list(self._models.keys())

    @property
    def loaded_teams(self) -> list[str]:
        """Veri yüklenmiş tüm takımlar (heuristic dahil)."""
        return list(self._data.keys())
