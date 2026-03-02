"""
decision_flow_gen.py – Interactive Decision Flowcharts.

Bot bir karar verdiğinde, bu karara hangi aşamalardan geçerek
vardığını bir Akış Şeması (Flowchart) olarak görselleştirir.

Örnek:
  Kadro Tamam ✅ → xG Yüksek ✅ → Oran Değerli ✅
  → Hava Uygun ⚠️ → Fuzzy Risk: Orta → KARAR: OYNA

Kavramlar:
  - Decision Node: Karar noktası (kare/daire)
  - Edge: Geçiş (ok)
  - Color Coding: Yeşil=geçti, sarı=uyarı, kırmızı=başarısız
  - Graphviz DOT: Graf tanımlama dili
  - Mermaid.js: Markdown uyumlu akış şemaları
  - Image Export: PNG/SVG olarak Telegram'a gönderme

Akış:
  1. Ensemble kararlarını topla (her modülün çıktısı)
  2. Karar ağacını oluştur (koşullar + sonuçlar)
  3. Graphviz ile PNG oluştur VEYA Mermaid metni üret
  4. Telegram'a resim olarak gönder

Teknoloji: Graphviz (graphviz Python bindings)
Fallback: Basit ASCII art + Mermaid.js text
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

try:
    import graphviz
    GV_OK = True
except ImportError:
    GV_OK = False
    logger.debug("graphviz yüklü değil – ASCII/Mermaid fallback.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
FLOW_DIR = ROOT / "data" / "flows"
FLOW_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class DecisionStep:
    """Karar akışındaki bir adım."""
    name: str = ""              # "Kadro Kontrolü"
    result: str = ""            # "TAMAM" | "UYARI" | "BAŞARISIZ"
    value: str = ""             # "11/11 fit"
    status: str = "pass"       # "pass" | "warn" | "fail"
    module: str = ""            # Kaynak modül


@dataclass
class DecisionFlow:
    """Tam karar akışı."""
    match_id: str = ""
    home_team: str = ""
    away_team: str = ""
    steps: list[DecisionStep] = field(default_factory=list)
    final_decision: str = ""    # "OYNA" | "PAS GEÇ" | "YARI KELLY"
    confidence: float = 0.0
    ev_pct: float = 0.0


# ═══════════════════════════════════════════════
#  RENK KODLARI
# ═══════════════════════════════════════════════
COLORS = {
    "pass": {"bg": "#2ECC71", "border": "#27AE60", "text": "#FFFFFF", "emoji": "✅"},
    "warn": {"bg": "#F39C12", "border": "#E67E22", "text": "#FFFFFF", "emoji": "⚠️"},
    "fail": {"bg": "#E74C3C", "border": "#C0392B", "text": "#FFFFFF", "emoji": "❌"},
    "decision": {"bg": "#3498DB", "border": "#2980B9", "text": "#FFFFFF"},
}

DECISION_COLORS = {
    "OYNA": "#2ECC71",
    "PAS GEÇ": "#E74C3C",
    "YARI KELLY": "#F39C12",
}


# ═══════════════════════════════════════════════
#  DECISION FLOW GENERATOR (Ana Sınıf)
# ═══════════════════════════════════════════════
class DecisionFlowGenerator:
    """Karar akış şeması üretici.

    Kullanım:
        dfg = DecisionFlowGenerator()

        flow = DecisionFlow(
            match_id="gs_fb",
            home_team="Galatasaray",
            away_team="Fenerbahçe",
            steps=[
                DecisionStep("Kadro Kontrolü", "11/11 fit", "TAMAM", "pass", "lineup"),
                DecisionStep("xG Analizi", "xG=1.8", "YÜKSEK", "pass", "poisson"),
                ...
            ],
            final_decision="OYNA",
            confidence=0.72,
            ev_pct=8.5,
        )

        # Görsel oluştur
        path = dfg.generate_image(flow)

        # Telegram metni
        text = dfg.generate_text(flow)

        # Mermaid.js
        mermaid = dfg.generate_mermaid(flow)
    """

    def __init__(self):
        logger.debug(
            f"[FlowGen] Başlatıldı (graphviz={'OK' if GV_OK else 'fallback'})"
        )

    def generate_image(self, flow: DecisionFlow,
                         filename: str = "") -> str:
        """PNG akış şeması oluştur."""
        if not filename:
            filename = f"flow_{flow.match_id}_{int(time.time())}"

        if GV_OK:
            return self._generate_graphviz(flow, filename)
        if MPL_OK:
            return self._generate_matplotlib(flow, filename)
        return self._generate_ascii_file(flow, filename)

    def _generate_graphviz(self, flow: DecisionFlow,
                              filename: str) -> str:
        """Graphviz ile görsel."""
        dot = graphviz.Digraph(
            name=filename,
            format="png",
            graph_attr={
                "rankdir": "TB",
                "bgcolor": "#1a1a2e",
                "fontname": "Arial",
                "pad": "0.5",
            },
            node_attr={
                "fontname": "Arial",
                "fontsize": "11",
                "style": "filled,rounded",
                "shape": "box",
                "margin": "0.2",
            },
            edge_attr={
                "color": "#CCCCCC",
                "penwidth": "1.5",
                "arrowsize": "0.8",
            },
        )

        # Başlık
        dot.node(
            "title",
            f"{flow.home_team} vs {flow.away_team}",
            shape="ellipse",
            fillcolor="#16213e",
            fontcolor="#FFFFFF",
            fontsize="14",
        )

        # Adımlar
        prev_id = "title"
        for i, step in enumerate(flow.steps):
            node_id = f"step_{i}"
            c = COLORS.get(step.status, COLORS["pass"])
            label = f"{c['emoji']} {step.name}\n{step.value}"

            dot.node(
                node_id,
                label,
                fillcolor=c["bg"],
                fontcolor=c["text"],
                color=c["border"],
            )
            dot.edge(prev_id, node_id)
            prev_id = node_id

        # Final karar
        dec_color = DECISION_COLORS.get(flow.final_decision, "#3498DB")
        dot.node(
            "decision",
            f"KARAR: {flow.final_decision}\n"
            f"Güven: {flow.confidence:.0%} | EV: {flow.ev_pct:+.1f}%",
            shape="diamond",
            fillcolor=dec_color,
            fontcolor="#FFFFFF",
            fontsize="13",
        )
        dot.edge(prev_id, "decision")

        # Render
        path = str(FLOW_DIR / filename)
        try:
            dot.render(path, cleanup=True)
            output = f"{path}.png"
            logger.info(f"[FlowGen] Graphviz PNG: {output}")
            return output
        except Exception as e:
            logger.debug(f"[FlowGen] Graphviz render hatası: {e}")
            if MPL_OK:
                return self._generate_matplotlib(flow, filename)
            return self._generate_ascii_file(flow, filename)

    def _generate_matplotlib(self, flow: DecisionFlow,
                                filename: str) -> str:
        """Matplotlib ile basit akış şeması."""
        n_steps = len(flow.steps) + 2  # başlık + karar
        fig, ax = plt.subplots(figsize=(6, n_steps * 0.8 + 1))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, n_steps + 1)
        ax.axis("off")
        fig.patch.set_facecolor("#1a1a2e")

        y = n_steps

        # Başlık
        ax.text(5, y, f"{flow.home_team} vs {flow.away_team}",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#16213e",
                          edgecolor="#CCCCCC"))
        y -= 1

        # Adımlar
        for step in flow.steps:
            c = COLORS.get(step.status, COLORS["pass"])
            emoji = c["emoji"]
            ax.text(5, y, f"{emoji} {step.name}: {step.value}",
                    ha="center", va="center", fontsize=10,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=c["bg"], edgecolor=c["border"]))
            ax.annotate("", xy=(5, y - 0.3), xytext=(5, y + 0.3 - 1),
                        arrowprops=dict(arrowstyle="->", color="#CCCCCC"))
            y -= 1

        # Final
        dec_color = DECISION_COLORS.get(flow.final_decision, "#3498DB")
        ax.text(5, y,
                f"KARAR: {flow.final_decision}\n"
                f"Güven: {flow.confidence:.0%} | EV: {flow.ev_pct:+.1f}%",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.5",
                          facecolor=dec_color, edgecolor="white", lw=2))

        path = FLOW_DIR / f"{filename}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"[FlowGen] Matplotlib PNG: {path}")
        return str(path)

    def _generate_ascii_file(self, flow: DecisionFlow,
                                filename: str) -> str:
        """ASCII fallback."""
        text = self.generate_text(flow)
        path = FLOW_DIR / f"{filename}.txt"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def generate_text(self, flow: DecisionFlow) -> str:
        """Telegram metin formatı."""
        lines = [
            f"📋 <b>KARAR AKIŞI: {flow.home_team} vs {flow.away_team}</b>\n",
            "─" * 35,
        ]

        for i, step in enumerate(flow.steps, 1):
            emoji = COLORS.get(step.status, COLORS["pass"])["emoji"]
            lines.append(f"  {i}. {emoji} {step.name}: {step.value}")

        lines.extend([
            "",
            "─" * 35,
            f"🎯 <b>KARAR: {flow.final_decision}</b>",
            f"📊 Güven: {flow.confidence:.0%} | EV: {flow.ev_pct:+.1f}%",
        ])

        return "\n".join(lines)

    def generate_mermaid(self, flow: DecisionFlow) -> str:
        """Mermaid.js formatı."""
        lines = ["graph TD"]
        lines.append(
            f'    A["{flow.home_team} vs {flow.away_team}"]'
        )

        prev = "A"
        for i, step in enumerate(flow.steps):
            node = chr(66 + i)  # B, C, D, ...
            emoji = COLORS.get(step.status, COLORS["pass"])["emoji"]
            lines.append(
                f'    {prev} --> {node}["{emoji} {step.name}\\n{step.value}"]'
            )
            # Stil
            color = COLORS.get(step.status, COLORS["pass"])["bg"]
            lines.append(
                f"    style {node} fill:{color},color:#fff"
            )
            prev = node

        # Karar
        dec = chr(66 + len(flow.steps))
        dec_color = DECISION_COLORS.get(flow.final_decision, "#3498DB")
        lines.append(
            f'    {prev} --> {dec}{{{{"KARAR: {flow.final_decision}\\n'
            f'Güven: {flow.confidence:.0%}"}}}}'
        )
        lines.append(f"    style {dec} fill:{dec_color},color:#fff")

        return "\n".join(lines)

    def build_flow_from_analysis(self, analysis: dict,
                                    match_id: str = "",
                                    home: str = "",
                                    away: str = "") -> DecisionFlow:
        """Analiz sonuçlarından otomatik akış oluştur."""
        flow = DecisionFlow(
            match_id=match_id,
            home_team=home,
            away_team=away,
        )

        # Standart adımlar (analiz dict'inden)
        checks = [
            ("Kadro Kontrolü", "lineup_ok", "lineup"),
            ("xG Analizi", "xg_score", "poisson"),
            ("Oran Değeri (EV)", "ev_pct", "fair_value"),
            ("Form Analizi", "form_score", "lstm"),
            ("Hava Durumu", "weather_ok", "fuzzy"),
            ("Sağkalım Analizi", "survival_ok", "survival"),
            ("Yorgunluk Kontrolü", "fatigue_ok", "fatigue"),
            ("Kaos Filtresi", "chaos_ok", "chaos"),
            ("Belirsizlik", "uncertainty_ok", "uncertainty"),
            ("Anomali Taraması", "anomaly_ok", "topology"),
            ("Kriz Analizi", "crisis_ok", "graphrag"),
        ]

        for name, key, module in checks:
            val = analysis.get(key)
            if val is None:
                continue

            if isinstance(val, bool):
                status = "pass" if val else "fail"
                display = "TAMAM" if val else "SORUNLU"
            elif isinstance(val, (int, float)):
                if key == "ev_pct":
                    status = "pass" if val > 0 else ("warn" if val > -3 else "fail")
                    display = f"{val:+.1f}%"
                elif key in ("xg_score", "form_score"):
                    status = "pass" if val > 0.5 else ("warn" if val > 0.3 else "fail")
                    display = f"{val:.2f}"
                else:
                    status = "pass" if val > 0.5 else "warn"
                    display = f"{val:.2f}"
            else:
                status = "pass"
                display = str(val)

            flow.steps.append(DecisionStep(
                name=name, result=display, value=display,
                status=status, module=module,
            ))

        # Final karar
        n_pass = sum(1 for s in flow.steps if s.status == "pass")
        n_fail = sum(1 for s in flow.steps if s.status == "fail")
        total = len(flow.steps) or 1

        if n_fail > 0:
            flow.final_decision = "PAS GEÇ"
        elif n_pass / total > 0.7:
            flow.final_decision = "OYNA"
        else:
            flow.final_decision = "YARI KELLY"

        flow.confidence = round(n_pass / total, 2)
        flow.ev_pct = analysis.get("ev_pct", 0.0)

        return flow
