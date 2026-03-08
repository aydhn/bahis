"""
mimic_engine.py – Adversarial Persona Mimicry (Anti-Ban Sistemi).

Bahis sitelerinin güvenlik algoritmaları (Cloudflare Turnstile vb.)
"Mouse hareketi olmayan" veya "Doğrusal tıklama yapan" trafiği
engeller. Bot insana benzemelidir.

Taklit Profilleri:
  - TiredHuman: Yavaş, düzensiz, duraklamalar
  - ExcitedHuman: Hızlı ama sinirli, geri dönüşler
  - CasualBrowser: Orta tempo, meraklı tarama
  - ProfessionalAnalyst: Sistematik ama doğal

Teknikler:
  1. Bezier Curve Mouse Movement: Fare eğrisel hareket
  2. Hesitation Latency: Butona git, dur, geri çek, bas
  3. Scroll Noise: İnsan benzeri kaydırma davranışı
  4. Typing Rhythm: Tuş basış aralıkları (keystroke dynamics)
  5. Session Fingerprint: Her oturumda farklı davranış profili
  6. Circadian Rhythm: Günün saatine göre davranış hızı

Teknoloji: numpy (B-Spline), random, asyncio
Entegrasyon: stealth_browser, api_hijacker
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class PersonaType(str, Enum):
    TIRED = "tired_human"
    EXCITED = "excited_human"
    CASUAL = "casual_browser"
    PROFESSIONAL = "professional_analyst"
    RANDOM = "random"


@dataclass
class PersonaProfile:
    """İnsan davranış profili."""
    name: str = ""
    persona_type: PersonaType = PersonaType.CASUAL
    # Hız parametreleri (saniye)
    min_action_delay: float = 0.3
    max_action_delay: float = 2.0
    avg_typing_interval: float = 0.12      # Tuş arası (saniye)
    typing_variance: float = 0.05
    # Fare parametreleri
    mouse_speed: float = 1.0               # 0.5=yavaş, 2.0=hızlı
    mouse_jitter: float = 2.0              # Piksel gürültü
    bezier_control_points: int = 3         # Eğri karmaşıklığı
    hesitation_prob: float = 0.15          # Tereddüt olasılığı
    overshoot_prob: float = 0.10           # Hedefi aşma olasılığı
    # Scroll parametreleri
    scroll_speed: float = 1.0
    scroll_pause_prob: float = 0.20
    # Oturum davranışı
    pages_before_action: int = 2           # Aksiyon öncesi sayfa tarama
    idle_check_prob: float = 0.05          # Rastgele boşta kalma


# ═══════════════════════════════════════════════
#  PROFİL PRESETS
# ═══════════════════════════════════════════════
PERSONA_PRESETS: dict[PersonaType, PersonaProfile] = {
    PersonaType.TIRED: PersonaProfile(
        name="Yorgun Kullanıcı",
        persona_type=PersonaType.TIRED,
        min_action_delay=1.0, max_action_delay=4.0,
        avg_typing_interval=0.18, typing_variance=0.10,
        mouse_speed=0.6, mouse_jitter=4.0,
        bezier_control_points=4, hesitation_prob=0.30,
        overshoot_prob=0.15, scroll_speed=0.5,
        scroll_pause_prob=0.40, pages_before_action=3,
        idle_check_prob=0.10,
    ),
    PersonaType.EXCITED: PersonaProfile(
        name="Heyecanlı Bahisçi",
        persona_type=PersonaType.EXCITED,
        min_action_delay=0.1, max_action_delay=0.8,
        avg_typing_interval=0.08, typing_variance=0.04,
        mouse_speed=1.8, mouse_jitter=3.0,
        bezier_control_points=2, hesitation_prob=0.05,
        overshoot_prob=0.20, scroll_speed=2.0,
        scroll_pause_prob=0.05, pages_before_action=1,
        idle_check_prob=0.02,
    ),
    PersonaType.CASUAL: PersonaProfile(
        name="Rahat Tarayıcı",
        persona_type=PersonaType.CASUAL,
        min_action_delay=0.5, max_action_delay=2.5,
        avg_typing_interval=0.12, typing_variance=0.06,
        mouse_speed=1.0, mouse_jitter=2.0,
        bezier_control_points=3, hesitation_prob=0.15,
        overshoot_prob=0.08, scroll_speed=1.0,
        scroll_pause_prob=0.20, pages_before_action=2,
        idle_check_prob=0.05,
    ),
    PersonaType.PROFESSIONAL: PersonaProfile(
        name="Profesyonel Analist",
        persona_type=PersonaType.PROFESSIONAL,
        min_action_delay=0.3, max_action_delay=1.5,
        avg_typing_interval=0.10, typing_variance=0.03,
        mouse_speed=1.2, mouse_jitter=1.5,
        bezier_control_points=3, hesitation_prob=0.10,
        overshoot_prob=0.05, scroll_speed=1.3,
        scroll_pause_prob=0.15, pages_before_action=1,
        idle_check_prob=0.03,
    ),
}


# ═══════════════════════════════════════════════
#  BEZİER CURVE FARE HAREKETİ
# ═══════════════════════════════════════════════
def bezier_curve(start: tuple[float, float], end: tuple[float, float],
                  n_control: int = 3, n_points: int = 30,
                  jitter: float = 2.0) -> list[tuple[float, float]]:
    """Bezier eğrisi ile doğal fare hareketi üret.

    Doğrusal hareket yerine, insansı eğrisel yol.
    """
    sx, sy = start
    ex, ey = end

    # Rastgele kontrol noktaları (eğriyi "doğallaştırır")
    controls = [start]
    for i in range(n_control):
        t = (i + 1) / (n_control + 1)
        cx = sx + (ex - sx) * t + random.gauss(0, abs(ex - sx) * 0.15)
        cy = sy + (ey - sy) * t + random.gauss(0, abs(ey - sy) * 0.15)
        controls.append((cx, cy))
    controls.append(end)

    # De Casteljau algoritması ile Bezier noktaları
    points = []
    for step in range(n_points + 1):
        t = step / n_points
        pts = list(controls)
        while len(pts) > 1:
            new_pts = []
            for j in range(len(pts) - 1):
                x = (1 - t) * pts[j][0] + t * pts[j + 1][0]
                y = (1 - t) * pts[j][1] + t * pts[j + 1][1]
                new_pts.append((x, y))
            pts = new_pts

        # Jitter (mikro titreşim) ekle
        px = pts[0][0] + random.gauss(0, jitter)
        py = pts[0][1] + random.gauss(0, jitter)
        points.append((round(px, 1), round(py, 1)))

    return points


def generate_hesitation_path(target: tuple[float, float],
                               jitter: float = 5.0) -> list[tuple[float, float]]:
    """Tereddüt hareketi: hedefe git, dur, hafif geri çek, sonra bas."""
    tx, ty = target

    approach = (tx + random.uniform(-15, 15), ty + random.uniform(-15, 15))
    pullback = (tx + random.uniform(5, 20), ty + random.uniform(5, 20))

    path = [approach, approach, pullback, target, target]
    return path


# ═══════════════════════════════════════════════
#  YAZMA RİTMİ
# ═══════════════════════════════════════════════
def generate_typing_delays(text: str, avg_interval: float = 0.12,
                            variance: float = 0.05) -> list[float]:
    """İnsan benzeri tuş basış aralıkları.

    Aynı harfi hızlı, farklı harfi yavaş basar.
    Boşluk ve noktalama sonrası daha uzun bekleme.
    """
    delays = []
    prev_char = ""

    for char in text:
        base = avg_interval

        if char in " .,;:!?":
            base *= random.uniform(1.5, 3.0)
        elif char == prev_char:
            base *= random.uniform(0.6, 0.8)
        elif char.isupper():
            base *= random.uniform(1.2, 1.6)

        delay = max(0.02, random.gauss(base, variance))
        delays.append(round(delay, 4))
        prev_char = char

    return delays


# ═══════════════════════════════════════════════
#  CIRCADIAN RHYTHM
# ═══════════════════════════════════════════════
def circadian_speed_factor() -> float:
    """Günün saatine göre hız çarpanı.

    Sabah erken / gece geç → yavaş
    Öğle sonrası → hızlı (uyanık)
    """
    hour = time.localtime().tm_hour

    if 6 <= hour < 9:
        return random.uniform(0.7, 0.9)
    elif 9 <= hour < 12:
        return random.uniform(0.9, 1.1)
    elif 12 <= hour < 14:
        return random.uniform(0.85, 1.0)
    elif 14 <= hour < 18:
        return random.uniform(1.0, 1.2)
    elif 18 <= hour < 22:
        return random.uniform(0.9, 1.1)
    elif 22 <= hour or hour < 2:
        return random.uniform(0.6, 0.8)
    else:
        return random.uniform(0.5, 0.7)


# ═══════════════════════════════════════════════
#  MİMİK ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class MimicEngine:
    """İnsan davranışı taklit motoru (Anti-Ban).

    Kullanım:
        mimic = MimicEngine(persona="casual")
        # Fare hareketi üret
        path = mimic.mouse_path((100, 200), (500, 400))
        # Aksiyon gecikmesi
        await mimic.human_delay()
        # Yazma gecikmesi
        delays = mimic.typing_delays("Galatasaray vs Fenerbahçe")
        # Tereddüt
        if mimic.should_hesitate():
            await mimic.hesitate()
        # Scroll
        speed = mimic.scroll_amount()
        # Oturum parmak izi
        fp = mimic.session_fingerprint()
    """

    def __init__(self, persona: str | PersonaType = "random"):
        if isinstance(persona, str):
            try:
                persona = PersonaType(persona)
            except ValueError:
                persona = PersonaType.RANDOM

        if persona == PersonaType.RANDOM:
            persona = random.choice([
                PersonaType.TIRED, PersonaType.EXCITED,
                PersonaType.CASUAL, PersonaType.PROFESSIONAL,
            ])

        self._profile = PERSONA_PRESETS[persona]
        self._circadian = circadian_speed_factor()
        self._action_count = 0
        self._session_start = time.time()
        self._session_id = f"sess_{int(time.time())}_{random.randint(1000, 9999)}"

        logger.debug(
            f"[Mimic] Persona: {self._profile.name} "
            f"(circadian={self._circadian:.2f}x)"
        )

    @property
    def profile(self) -> PersonaProfile:
        return self._profile

    # ═══════════════════════════════════════════
    #  FARE HAREKETİ
    # ═══════════════════════════════════════════
    def mouse_path(self, start: tuple[float, float],
                    end: tuple[float, float],
                    n_points: int = 25) -> list[tuple[float, float]]:
        """Bezier eğrisi ile doğal fare yolu üret."""
        p = self._profile
        path = bezier_curve(
            start, end,
            n_control=p.bezier_control_points,
            n_points=n_points,
            jitter=p.mouse_jitter,
        )

        # Overshoot: %10 ihtimalle hedefi aş
        if random.random() < p.overshoot_prob:
            ex, ey = end
            overshoot = (
                ex + random.uniform(5, 25),
                ey + random.uniform(5, 25),
            )
            correction = bezier_curve(
                overshoot, end, n_control=1,
                n_points=5, jitter=1.0,
            )
            path.extend(correction)

        return path

    def mouse_step_delays(self, n_points: int) -> list[float]:
        """Her fare adımı arasındaki gecikme."""
        p = self._profile
        base = 0.01 / max(p.mouse_speed * self._circadian, 0.1)
        return [
            max(0.002, random.gauss(base, base * 0.3))
            for _ in range(n_points)
        ]

    # ═══════════════════════════════════════════
    #  GECİKME & TEREDDÜT
    # ═══════════════════════════════════════════
    async def human_delay(self, action: str = "click") -> float:
        """İnsan benzeri aksiyon gecikmesi."""
        p = self._profile
        base = random.uniform(p.min_action_delay, p.max_action_delay)
        delay = base / max(self._circadian, 0.1)

        # Yorgunluk: oturum uzadıkça yavaşla
        session_minutes = (time.time() - self._session_start) / 60
        fatigue = 1 + session_minutes * 0.005
        delay *= fatigue

        self._action_count += 1
        await asyncio.sleep(delay)
        return delay

    def should_hesitate(self) -> bool:
        """Tereddüt edecek mi?"""
        return random.random() < self._profile.hesitation_prob

    async def hesitate(self) -> float:
        """Tereddüt süresi (butona gidip duraksar)."""
        pause = random.uniform(0.3, 1.5) / max(self._circadian, 0.1)
        await asyncio.sleep(pause)
        return pause

    def hesitation_path(self, target: tuple[float, float]
                         ) -> list[tuple[float, float]]:
        """Tereddüt hareketi yolu."""
        return generate_hesitation_path(target, self._profile.mouse_jitter)

    # ═══════════════════════════════════════════
    #  YAZMA
    # ═══════════════════════════════════════════
    def typing_delays(self, text: str) -> list[float]:
        """İnsan benzeri yazma gecikmeleri."""
        p = self._profile
        delays = generate_typing_delays(
            text, p.avg_typing_interval, p.typing_variance,
        )
        return [d / max(self._circadian, 0.1) for d in delays]

    # ═══════════════════════════════════════════
    #  SCROLL
    # ═══════════════════════════════════════════
    def scroll_amount(self) -> int:
        """Scroll miktarı (piksel)."""
        base = random.randint(100, 400)
        return int(base * self._profile.scroll_speed * self._circadian)

    def should_pause_scroll(self) -> bool:
        """Scroll sırasında durak verecek mi?"""
        return random.random() < self._profile.scroll_pause_prob

    # ═══════════════════════════════════════════
    #  OTURUM DAVRANIŞI
    # ═══════════════════════════════════════════
    def should_browse_first(self) -> bool:
        """Aksiyondan önce rastgele sayfa gezecek mi?"""
        return self._action_count < self._profile.pages_before_action

    def should_idle(self) -> bool:
        """Rastgele boşta kalacak mı?"""
        return random.random() < self._profile.idle_check_prob

    async def idle_pause(self) -> float:
        """Boşta kalma süresi."""
        pause = random.uniform(3.0, 15.0)
        await asyncio.sleep(pause)
        return pause

    def session_fingerprint(self) -> dict:
        """Oturum parmak izi (her seferinde farklı)."""
        return {
            "session_id": self._session_id,
            "persona": self._profile.persona_type.value,
            "circadian": round(self._circadian, 3),
            "actions": self._action_count,
            "duration_min": round(
                (time.time() - self._session_start) / 60, 1,
            ),
            "viewport": (
                random.choice([1366, 1440, 1536, 1920, 2560]),
                random.choice([768, 900, 1024, 1080, 1440]),
            ),
            "timezone_offset": random.choice([-180, -120, 0, 60, 120, 180]),
            "language": random.choice(["tr-TR", "en-US", "tr"]),
            "platform": random.choice([
                "Win32", "Win64", "MacIntel", "Linux x86_64",
            ]),
        }

    def randomize_persona(self) -> None:
        """Yeni rastgele persona seç (oturum ortasında değişim)."""
        new_type = random.choice([
            PersonaType.TIRED, PersonaType.EXCITED,
            PersonaType.CASUAL, PersonaType.PROFESSIONAL,
        ])
        self._profile = PERSONA_PRESETS[new_type]
        self._circadian = circadian_speed_factor()
        logger.debug(f"[Mimic] Persona değişti → {self._profile.name}")
