"""
vision_tracker.py – YOLO + OpenCV ile canlı maç yayınından pozisyon takibi.
Oyuncu konumlarını ve top hareketini frame-by-frame analiz eder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV yüklü değil – VisionTracker devre dışı.")


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class FrameAnalysis:
    frame_no: int
    players: list[Detection] = field(default_factory=list)
    ball: Detection | None = None
    heatmap: np.ndarray | None = None


class VisionTracker:
    """YOLO tabanlı canlı maç görüntü analizi."""

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._heatmap: np.ndarray | None = None
        self._frame_count = 0

        if model_path:
            self._load_model(model_path)
        logger.debug("VisionTracker başlatıldı.")

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            logger.info(f"YOLO modeli yüklendi: {model_path}")
        except ImportError:
            logger.warning("ultralytics yüklü değil – YOLO devre dışı.")
        except Exception as e:
            logger.warning(f"Model yükleme hatası: {e}")

    def process_frame(self, frame: np.ndarray) -> FrameAnalysis:
        """Tek bir frame'i analiz eder."""
        if cv2 is None:
            return FrameAnalysis(frame_no=self._frame_count)

        self._frame_count += 1
        analysis = FrameAnalysis(frame_no=self._frame_count)

        if self._model is not None:
            results = self._model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]

                    det = Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2))

                    if label.lower() in ("person", "player"):
                        analysis.players.append(det)
                    elif label.lower() in ("ball", "sports ball"):
                        analysis.ball = det
        else:
            analysis = self._fallback_detect(frame)

        # Heatmap güncelle
        self._update_heatmap(frame.shape[:2], analysis)
        analysis.heatmap = self._heatmap.copy() if self._heatmap is not None else None

        return analysis

    def _fallback_detect(self, frame: np.ndarray) -> FrameAnalysis:
        """YOLO yoksa basit renk tabanlı tespit."""
        analysis = FrameAnalysis(frame_no=self._frame_count)
        if cv2 is None:
            return analysis

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Top tespiti (beyaz/sarı bölgeler)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                circularity = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
                if circularity > 0.6:
                    analysis.ball = Detection(
                        label="ball", confidence=circularity,
                        bbox=(x, y, x + w, y + h),
                    )
                    break

        return analysis

    def _update_heatmap(self, shape: tuple[int, int], analysis: FrameAnalysis):
        h, w = shape
        if self._heatmap is None:
            self._heatmap = np.zeros((h, w), dtype=np.float32)

        for player in analysis.players:
            cx, cy = int(player.center[0]), int(player.center[1])
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(self._heatmap, (cx, cy), 15, 1.0, -1)

        # Zamanla soldur
        self._heatmap *= 0.99

    def process_stream(self, stream_url: str, max_frames: int = 1000):
        """Video akışını işler."""
        if cv2 is None:
            logger.error("OpenCV gerekli – yüklenmemiş.")
            return

        cap = cv2.VideoCapture(stream_url)
        frames_processed = 0

        while cap.isOpened() and frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            analysis = self.process_frame(frame)
            frames_processed += 1

            if frames_processed % 100 == 0:
                logger.info(
                    f"Frame {frames_processed}: "
                    f"{len(analysis.players)} oyuncu, "
                    f"top={'var' if analysis.ball else 'yok'}"
                )

        cap.release()
        logger.info(f"Stream tamamlandı: {frames_processed} frame işlendi.")

    def get_heatmap(self) -> np.ndarray | None:
        return self._heatmap
