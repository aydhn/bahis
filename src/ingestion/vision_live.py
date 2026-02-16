"""
vision_live.py – Real-Time Computer Vision (YOLOv8 Canlı Maç Analizi).

API verisi gecikmelidir (30-60 sn), görüntü gerçektir.
Botunuza "Göz" veriyoruz.

Süreç:
  1. Canlı yayın ekranından 1 FPS kare al
  2. YOLOv8n (nano) ile top ve oyuncu tespiti
  3. Topun konumunu takip et: ceza sahası, orta alan, kale çizgisi
  4. Baskı İndeksi (Pressure Index) hesapla
  5. "GOL KOKUSU" sinyali oluştur (top sürekli rakip sahada)

Teknoloji:
  - ultralytics (YOLOv8 Nano – hafif, CPU'da çalışır)
  - opencv-python (kare alma ve işleme)
  - Fallback: gradient/hareket tabanlı analiz (modelsiz)

Event Bus entegrasyonu:
  Olaylar → event_bus → scraper, model, telegram hepsi dinler.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    logger.info("opencv-python yüklü değil – vision devre dışı.")

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    logger.info("ultralytics yüklü değil – heuristic vision aktif.")


ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models" / "vision"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# FIFA standart saha boyutları (piksel haritalama için)
PITCH_LENGTH = 105  # metre
PITCH_WIDTH = 68


# ═══════════════════════════════════════════════
#  VERİ MODELLERİ
# ═══════════════════════════════════════════════
@dataclass
class DetectedObject:
    """Tespit edilen nesne."""
    class_name: str = ""         # ball | player | goalkeeper | referee
    confidence: float = 0.0
    bbox: tuple = (0, 0, 0, 0)  # (x1, y1, x2, y2)
    center: tuple = (0, 0)      # (cx, cy) piksel
    pitch_pos: tuple = (0, 0)   # (x, y) saha koordinatı (0-105, 0-68)
    team: str = ""               # home | away | unknown


@dataclass
class FrameAnalysis:
    """Tek kare analiz sonucu."""
    frame_idx: int = 0
    timestamp: float = 0.0
    objects: list[DetectedObject] = field(default_factory=list)
    ball_position: tuple = (0, 0)       # Saha koordinatı
    ball_in_penalty_area: bool = False
    ball_side: str = "midfield"          # home_half | midfield | away_half | penalty_home | penalty_away
    home_players_detected: int = 0
    away_players_detected: int = 0
    pressure_index: float = 0.0          # 0-1 (1 = max baskı)
    momentum_signal: str = ""            # GOAL_SMELL | HIGH_PRESSURE | COUNTER | NEUTRAL


@dataclass
class MomentumReport:
    """Zaman penceresi momentum raporu."""
    match_id: str = ""
    window_seconds: int = 60
    frames_analyzed: int = 0
    avg_pressure: float = 0.0
    ball_in_opponent_half_pct: float = 0.0
    ball_in_penalty_area_pct: float = 0.0
    dominant_side: str = ""              # home | away | balanced
    signal: str = ""                     # GOAL_SMELL | HIGH_PRESSURE | NEUTRAL
    signal_strength: float = 0.0         # 0-1


# ═══════════════════════════════════════════════
#  YOLO DETECTOR
# ═══════════════════════════════════════════════
class YOLOFootballDetector:
    """YOLOv8 ile futbol nesnelerini tespit et."""

    # COCO class ID → futbol nesnesi mapping
    FOOTBALL_CLASSES = {
        0: "player",       # person
        32: "ball",        # sports ball
    }

    def __init__(self, model_size: str = "n"):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium)
        """
        self._model = None
        self._model_size = model_size

        if YOLO_OK:
            try:
                model_name = f"yolov8{model_size}.pt"
                self._model = YOLO(model_name)
                logger.info(f"[Vision] YOLOv8{model_size} yüklendi.")
            except Exception as e:
                logger.warning(f"[Vision] YOLO yükleme hatası: {e}")

    def detect(self, frame: np.ndarray,
               conf_threshold: float = 0.3) -> list[DetectedObject]:
        """Tek karede nesne tespiti."""
        if self._model is None:
            return self._heuristic_detect(frame)

        try:
            results = self._model(frame, conf=conf_threshold, verbose=False)

            objects = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in (0, 32):
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    class_name = self.FOOTBALL_CLASSES.get(cls_id, "unknown")

                    obj = DetectedObject(
                        class_name=class_name,
                        confidence=round(conf, 3),
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        center=(cx, cy),
                    )
                    objects.append(obj)

            return objects

        except Exception as e:
            logger.debug(f"[Vision] YOLO detect hatası: {e}")
            return []

    def _heuristic_detect(self, frame: np.ndarray) -> list[DetectedObject]:
        """YOLO yoksa renk tabanlı basit tespit."""
        if not CV2_OK:
            return []

        objects = []
        h, w = frame.shape[:2]

        # HSV'ye çevir
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yeşil alan tespiti (saha)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / (h * w)

        # Beyaz top tespiti
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 2000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    objects.append(DetectedObject(
                        class_name="ball",
                        confidence=0.5,
                        center=(cx, cy),
                    ))
                    break

        return objects


# ═══════════════════════════════════════════════
#  CANLI MAÇ İZLEYİCİ
# ═══════════════════════════════════════════════
class LiveMatchVision:
    """Canlı maç görüntü analizi motoru.

    Kullanım:
        vision = LiveMatchVision()
        # Ekran görüntüsünden analiz
        analysis = vision.analyze_frame(frame, match_id="GS_FB")
        # Momentum raporu
        report = vision.get_momentum_report("GS_FB")
        # Sürekli izleme (asyncio task)
        await vision.watch_stream(source, match_id, shutdown_event)
    """

    # Saha bölgeleri (normalize: 0-1 → 0-105)
    PENALTY_AREA_START = 88.5 / 105      # Rakip ceza sahası başlangıcı
    HALF_LINE = 52.5 / 105               # Orta çizgi

    # Eşikler
    GOAL_SMELL_THRESHOLD = 0.70          # Baskı endeksi > 0.70 → gol kokusu
    PENALTY_TIME_THRESHOLD = 30          # 30 sn ceza sahasında → sinyal
    PRESSURE_WINDOW = 60                 # 60 saniyelik pencere

    def __init__(self, model_size: str = "n",
                 fps: float = 1.0):
        self._detector = YOLOFootballDetector(model_size=model_size)
        self._fps = fps
        self._frame_buffer: dict[str, deque[FrameAnalysis]] = {}
        self._match_stats: dict[str, dict] = {}
        logger.debug(f"[Vision] LiveMatchVision başlatıldı (fps={fps}).")

    def analyze_frame(self, frame: np.ndarray,
                       match_id: str = "",
                       frame_idx: int = 0) -> FrameAnalysis:
        """Tek kareyi analiz et."""
        analysis = FrameAnalysis(
            frame_idx=frame_idx,
            timestamp=time.time(),
        )

        if frame is None or frame.size == 0:
            return analysis

        h, w = frame.shape[:2]

        # Nesne tespiti
        objects = self._detector.detect(frame)
        analysis.objects = objects

        # Top pozisyonu
        balls = [o for o in objects if o.class_name == "ball"]
        if balls:
            ball = max(balls, key=lambda b: b.confidence)
            # Piksel → saha koordinatı (basit linear mapping)
            pitch_x = (ball.center[0] / w) * PITCH_LENGTH
            pitch_y = (ball.center[1] / h) * PITCH_WIDTH
            analysis.ball_position = (round(pitch_x, 1), round(pitch_y, 1))

            # Bölge tespiti
            norm_x = ball.center[0] / w
            if norm_x > self.PENALTY_AREA_START:
                analysis.ball_side = "penalty_away"
                analysis.ball_in_penalty_area = True
            elif norm_x > self.HALF_LINE:
                analysis.ball_side = "away_half"
            elif norm_x < (1.0 - self.PENALTY_AREA_START):
                analysis.ball_side = "penalty_home"
                analysis.ball_in_penalty_area = True
            else:
                analysis.ball_side = "home_half"

        # Oyuncu sayıları (basit: sol yarı = home, sağ yarı = away)
        players = [o for o in objects if o.class_name == "player"]
        for p in players:
            if p.center[0] < w / 2:
                analysis.home_players_detected += 1
            else:
                analysis.away_players_detected += 1

        # Baskı indeksi hesapla
        analysis.pressure_index = self._calculate_pressure(analysis, objects, w, h)

        # Momentum sinyali
        analysis.momentum_signal = self._classify_momentum(analysis, match_id)

        # Buffer'a ekle
        if match_id not in self._frame_buffer:
            self._frame_buffer[match_id] = deque(maxlen=int(self.PRESSURE_WINDOW * self._fps))
        self._frame_buffer[match_id].append(analysis)

        return analysis

    def _calculate_pressure(self, analysis: FrameAnalysis,
                              objects: list[DetectedObject],
                              frame_w: int, frame_h: int) -> float:
        """Baskı endeksi hesapla (0-1)."""
        pressure = 0.0

        # Top konumu faktörü (rakip sahada = yüksek baskı)
        if analysis.ball_side == "penalty_away":
            pressure += 0.5
        elif analysis.ball_side == "away_half":
            pressure += 0.3
        elif analysis.ball_side == "midfield":
            pressure += 0.1

        # Oyuncu yoğunluğu: rakip yarıdaki oyuncu sayısı
        players_in_opponent = sum(
            1 for o in objects
            if o.class_name == "player" and o.center[0] > frame_w * 0.5
        )
        if players_in_opponent > 5:
            pressure += 0.3
        elif players_in_opponent > 3:
            pressure += 0.15

        # Top + çok oyuncu = yüksek baskı
        if analysis.ball_in_penalty_area and players_in_opponent > 4:
            pressure += 0.2

        return min(pressure, 1.0)

    def _classify_momentum(self, analysis: FrameAnalysis,
                            match_id: str) -> str:
        """Momentum sinyalini sınıfla."""
        buf = self._frame_buffer.get(match_id, deque())

        if len(buf) < 5:
            return "NEUTRAL"

        # Son N karedeki ortalama baskı
        recent = list(buf)[-int(self.PRESSURE_WINDOW * self._fps):]
        avg_pressure = np.mean([f.pressure_index for f in recent])

        # Ceza sahası süresi
        penalty_frames = sum(1 for f in recent if f.ball_in_penalty_area)
        penalty_pct = penalty_frames / max(len(recent), 1)

        if avg_pressure > self.GOAL_SMELL_THRESHOLD and penalty_pct > 0.4:
            return "GOAL_SMELL"
        elif avg_pressure > 0.5:
            return "HIGH_PRESSURE"
        elif avg_pressure < 0.15:
            return "COUNTER"

        return "NEUTRAL"

    # ═══════════════════════════════════════════
    #  MOMENTUM RAPORU
    # ═══════════════════════════════════════════
    def get_momentum_report(self, match_id: str) -> MomentumReport:
        """Son N saniyelik momentum raporu."""
        buf = self._frame_buffer.get(match_id, deque())
        report = MomentumReport(
            match_id=match_id,
            window_seconds=self.PRESSURE_WINDOW,
        )

        if not buf:
            report.signal = "NO_DATA"
            return report

        frames = list(buf)
        report.frames_analyzed = len(frames)

        # Ortalama baskı
        pressures = [f.pressure_index for f in frames]
        report.avg_pressure = round(float(np.mean(pressures)), 3)

        # Rakip sahada bulunma yüzdesi
        away_half = sum(
            1 for f in frames
            if f.ball_side in ("away_half", "penalty_away")
        )
        report.ball_in_opponent_half_pct = round(away_half / len(frames), 3)

        # Ceza sahasında bulunma yüzdesi
        penalty = sum(1 for f in frames if f.ball_in_penalty_area)
        report.ball_in_penalty_area_pct = round(penalty / len(frames), 3)

        # Dominant taraf
        home_frames = sum(
            1 for f in frames if f.ball_side in ("home_half", "penalty_home")
        )
        away_frames = sum(
            1 for f in frames if f.ball_side in ("away_half", "penalty_away")
        )
        if away_frames > home_frames * 1.5:
            report.dominant_side = "home"  # Ev sahibi baskı yapıyor
        elif home_frames > away_frames * 1.5:
            report.dominant_side = "away"
        else:
            report.dominant_side = "balanced"

        # Sinyal
        if report.avg_pressure > self.GOAL_SMELL_THRESHOLD:
            report.signal = "GOAL_SMELL"
            report.signal_strength = min(report.avg_pressure / 0.9, 1.0)
        elif report.avg_pressure > 0.5:
            report.signal = "HIGH_PRESSURE"
            report.signal_strength = report.avg_pressure
        else:
            report.signal = "NEUTRAL"
            report.signal_strength = report.avg_pressure

        return report

    # ═══════════════════════════════════════════
    #  SÜREKLİ İZLEME
    # ═══════════════════════════════════════════
    async def watch_stream(self, source: str | int,
                            match_id: str,
                            shutdown: asyncio.Event,
                            event_bus: Any = None,
                            notifier: Any = None):
        """Video akışını sürekli izle ve sinyal üret.

        Args:
            source: Video URL, dosya yolu veya kamera index
            match_id: Maç tanımlayıcı
            shutdown: Kapatma sinyali
            event_bus: Olay veri yolu
            notifier: Telegram bildirim servisi
        """
        if not CV2_OK:
            logger.warning("[Vision] OpenCV yüklü değil – izleme devre dışı.")
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"[Vision] Kaynak açılamadı: {source}")
            return

        logger.info(f"[Vision] Canlı izleme başladı: {match_id} ({source})")
        frame_idx = 0
        last_signal = ""
        interval = 1.0 / self._fps

        try:
            while not shutdown.is_set():
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(1)
                    continue

                analysis = self.analyze_frame(frame, match_id, frame_idx)
                frame_idx += 1

                # Yeni sinyal varsa bildir
                if (analysis.momentum_signal != last_signal and
                    analysis.momentum_signal in ("GOAL_SMELL", "HIGH_PRESSURE")):

                    last_signal = analysis.momentum_signal
                    report = self.get_momentum_report(match_id)

                    logger.warning(
                        f"[Vision] {match_id}: {analysis.momentum_signal} "
                        f"(baskı={report.avg_pressure:.0%}, "
                        f"ceza_sahası={report.ball_in_penalty_area_pct:.0%})"
                    )

                    # Event Bus'a yayınla
                    if event_bus:
                        from src.core.event_bus import Event
                        await event_bus.emit(Event(
                            event_type="vision_signal",
                            source="vision_live",
                            match_id=match_id,
                            data={
                                "signal": analysis.momentum_signal,
                                "pressure": report.avg_pressure,
                                "ball_position": analysis.ball_position,
                                "penalty_area_pct": report.ball_in_penalty_area_pct,
                            },
                        ))

                    # Telegram bildirimi
                    if notifier and analysis.momentum_signal == "GOAL_SMELL":
                        emoji = "👃🔥" if report.avg_pressure > 0.8 else "👃"
                        await notifier.send(
                            f"{emoji} <b>GOL KOKUSU:</b> {match_id}\n"
                            f"Baskı: {report.avg_pressure:.0%}\n"
                            f"Ceza Sahasında: {report.ball_in_penalty_area_pct:.0%}\n"
                            f"Dominant: {report.dominant_side}"
                        )

                elif analysis.momentum_signal == "NEUTRAL":
                    last_signal = ""

                await asyncio.sleep(interval)

        finally:
            cap.release()
            logger.info(f"[Vision] İzleme durduruldu: {match_id}")

    # ═══════════════════════════════════════════
    #  EKRAN GÖRÜNTÜSÜNDEN ANALİZ
    # ═══════════════════════════════════════════
    def analyze_screenshot(self, image_path: str | Path,
                            match_id: str = "") -> FrameAnalysis:
        """Ekran görüntüsünden tek kare analizi."""
        if not CV2_OK:
            return FrameAnalysis()

        frame = cv2.imread(str(image_path))
        if frame is None:
            logger.warning(f"[Vision] Görüntü okunamadı: {image_path}")
            return FrameAnalysis()

        return self.analyze_frame(frame, match_id)

    def extract_player_coords(self, frame: np.ndarray) -> list[tuple[float, float]]:
        """Karedeki oyuncu koordinatlarını çıkar (TDA için)."""
        objects = self._detector.detect(frame)
        h, w = frame.shape[:2]
        coords = []
        for o in objects:
            if o.class_name == "player":
                px = (o.center[0] / w) * PITCH_LENGTH
                py = (o.center[1] / h) * PITCH_WIDTH
                coords.append((round(px, 1), round(py, 1)))
        return coords

    @property
    def active_matches(self) -> list[str]:
        return list(self._frame_buffer.keys())
