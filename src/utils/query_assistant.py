"""
query_assistant.py – Doğal Dil ile Veri Sorgulama (Text-to-SQL).

SQL veya Cypher sorgusu yazmakla uğraşmayın. Bota Türkçe sorun,
o veritabanını sorgulasın.

Siz: "Son 3 yılda yağmurlu havada Beşiktaş'ın kazanma oranı nedir?"
Bot: Bu cümleyi SQL'e çevirir → Çalıştırır → Sonucu özetler.

Teknoloji:
  - LangChain SQL Database Chain
  - Google Gemini API (Ücretsiz Tier) veya basit şablon eşleştirme
  - DuckDB (ana veritabanı)

Fallback: LLM yoksa regex + template bazlı SQL üreteci
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import duckdb
    DUCK_OK = True
except ImportError:
    DUCK_OK = False

try:
    from langchain_community.utilities import SQLDatabase
    from langchain.chains import create_sql_query_chain
    LANGCHAIN_SQL_OK = True
except ImportError:
    LANGCHAIN_SQL_OK = False

try:
    # import google.generativeai as genai
    GEMINI_OK = False
except ImportError:
    GEMINI_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class QueryResult:
    """Sorgu sonucu."""
    question: str = ""
    sql: str = ""
    raw_result: Any = None
    formatted_answer: str = ""
    execution_time_ms: float = 0.0
    method: str = ""              # gemini | langchain | template | error
    confidence: float = 0.0
    error: str = ""


# ═══════════════════════════════════════════════
#  ŞABLON BAZLI SQL ÜRETECİ (LLM Fallback)
# ═══════════════════════════════════════════════
class TemplateSQLGenerator:
    """Regex + şablon bazlı SQL üreteci.

    LLM/API olmadan basit Türkçe soruları SQL'e çevirir.
    """

    # Takım eşleştirme
    TEAM_ALIASES = {
        "galatasaray": ["gs", "galatasaray", "cimbom", "aslan"],
        "fenerbahçe": ["fb", "fenerbahçe", "fenerbahce", "kanarya", "fener"],
        "beşiktaş": ["bjk", "beşiktaş", "besiktas", "kartal"],
        "trabzonspor": ["ts", "trabzonspor", "trabzon", "bordo mavi"],
        "başakşehir": ["basaksehir", "başakşehir", "ibfk"],
        "adana demirspor": ["adana", "adana demir"],
        "antalyaspor": ["antalya", "antalyaspor"],
        "konyaspor": ["konya", "konyaspor"],
        "sivasspor": ["sivas", "sivasspor"],
        "alanyaspor": ["alanya", "alanyaspor"],
        "kayserispor": ["kayseri", "kayserispor"],
        "kasımpaşa": ["kasimpasa", "kasımpaşa"],
        "hatayspor": ["hatay", "hatayspor"],
        "pendikspor": ["pendik", "pendikspor"],
        "gaziantep fk": ["gaziantep", "gaziantepfk"],
        "samsunspor": ["samsun", "samsunspor"],
        "rizespor": ["rize", "rizespor"],
        "istanbulspor": ["istanbulspor"],
    }

    # Soru şablonları
    PATTERNS = [
        {
            "regex": r"(?:son\s+)?(\d+)\s+(?:yıl|sezon).*?(\w+).*?(?:kazanma|galibiyet)\s*(?:oranı|yüzde)",
            "sql": "SELECT COUNT(*) FILTER (WHERE result='W') * 100.0 / COUNT(*) as win_pct FROM matches WHERE team='{team}' AND season >= {year}",
            "description": "Kazanma oranı",
        },
        {
            "regex": r"(\w+)\s+(?:ile|vs|karşı)\s+(\w+).*?(?:son\s+)?(\d+)?\s*maç",
            "sql": "SELECT * FROM matches WHERE (home_team='{team1}' AND away_team='{team2}') OR (home_team='{team2}' AND away_team='{team1}') ORDER BY date DESC LIMIT {limit}",
            "description": "H2H maçlar",
        },
        {
            "regex": r"(?:en\s+(?:çok|fazla))\s+(?:gol)\s+(?:atan|olan)\s+(?:takım|takımlar)",
            "sql": "SELECT team, SUM(goals_scored) as total_goals FROM matches GROUP BY team ORDER BY total_goals DESC LIMIT 10",
            "description": "En golcü takımlar",
        },
        {
            "regex": r"(\w+).*?(?:son\s+)?(\d+)\s+maç.*?(?:gol|skor)",
            "sql": "SELECT date, opponent, goals_scored, goals_conceded FROM matches WHERE team='{team}' ORDER BY date DESC LIMIT {limit}",
            "description": "Son maç golleri",
        },
        {
            "regex": r"(?:alt|üst)\s+(\d+\.?\d*)\s+(?:gol)?.*?(\w+)",
            "sql": "SELECT COUNT(*) FILTER (WHERE total_goals > {threshold}) * 100.0 / COUNT(*) as over_pct FROM matches WHERE team='{team}'",
            "description": "Alt/Üst yüzdesi",
        },
        {
            "regex": r"(?:ev\s+(?:sahibi|sahibinde)|deplasman).*?(\w+).*?(?:kazan|galip)",
            "sql": "SELECT COUNT(*) FILTER (WHERE result='W') * 100.0 / COUNT(*) as pct FROM matches WHERE team='{team}' AND venue='{venue}'",
            "description": "Ev/deplasman performansı",
        },
    ]

    def find_team(self, text: str) -> str | None:
        """Metinden takım ismi çıkar."""
        text_lower = text.lower()
        for canonical, aliases in self.TEAM_ALIASES.items():
            for alias in aliases:
                if alias in text_lower:
                    return canonical
        return None

    def generate(self, question: str) -> str | None:
        """Şablon eşleştirme ile SQL üret."""
        q = question.lower().strip()

        for pattern in self.PATTERNS:
            match = re.search(pattern["regex"], q, re.IGNORECASE)
            if match:
                sql = pattern["sql"]
                groups = match.groups()

                team = self.find_team(question)
                if team:
                    sql = sql.replace("{team}", team)
                    sql = sql.replace("{team1}", team)

                for g in groups:
                    if g and g.isdigit():
                        sql = sql.replace("{limit}", g)
                        sql = sql.replace("{year}", str(2026 - int(g)))
                        sql = sql.replace("{threshold}", g)

                # Varsayılan değerler
                sql = sql.replace("{limit}", "10")
                sql = sql.replace("{year}", "2023")
                sql = sql.replace("{venue}", "home")
                sql = sql.replace("{threshold}", "2.5")

                return sql

        return None


# ═══════════════════════════════════════════════
#  GEMINI SQL ÜRETECİ
# ═══════════════════════════════════════════════
class GeminiSQLGenerator:
    """Google Gemini ile doğal dil → SQL."""

    SYSTEM_PROMPT = """Sen bir SQL uzmanısın. Kullanıcının Türkçe sorusunu DuckDB SQL sorgusuna çevir.

Veritabanı şeması:
- matches(date, home_team, away_team, home_goals, away_goals, league, season, venue, weather, referee)
- players(name, team, position, rating, goals, assists, minutes_played)
- odds(match_id, bookmaker, home_odds, draw_odds, away_odds, timestamp)
- predictions(match_id, model, prob_home, prob_draw, prob_away, prediction, result)

Sadece SQL sorgusu döndür, açıklama ekleme. Tehlikeli (DROP, DELETE, UPDATE) sorgu yazma."""

    def __init__(self):
        self._model = None
        if GEMINI_OK:
            try:
                self._model = None # Gemini Disabled
            except Exception:
                pass

    def generate(self, question: str) -> str | None:
        """Gemini ile SQL üret."""
        if not self._model:
            return None

        try:
            response = self._model.generate_content(
                f"{self.SYSTEM_PROMPT}\n\nSoru: {question}",
            )
            sql = response.text.strip()
            # SQL temizle
            sql = sql.replace("```sql", "").replace("```", "").strip()
            # Güvenlik kontrolü - DuckDB read_only mode still allows reading arbitrary files
            # via read_csv, read_parquet etc. Block these explicitly.
            dangerous_exact = [
                "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
                "COPY", "ATTACH", "DETACH", "PRAGMA", "CALL", "INSTALL", "LOAD", "SYSTEM"
            ]

            # Use regex with word boundaries to avoid matching substrings like "created_at".
            # Also block any function starting with "read_" (e.g., read_csv_auto, read_text)
            pattern = re.compile(
                rf"\b({'|'.join(dangerous_exact)})\b|\bread_\w*\b",
                re.IGNORECASE
            )

            if pattern.search(sql):
                logger.warning(f"[Security] Blocked potentially dangerous SQL: {sql}")
                return None
            return sql
        except Exception as e:
            logger.debug(f"[Query] Gemini hatası: {e}")
            return None


# ═══════════════════════════════════════════════
#  QUERY ASSISTANT (Ana Sınıf)
# ═══════════════════════════════════════════════
class QueryAssistant:
    """Doğal dil ile veritabanı sorgulama asistanı.

    Kullanım:
        assistant = QueryAssistant()
        result = assistant.ask("Son 3 yılda Beşiktaş'ın kazanma oranı nedir?")
        print(result.formatted_answer)

        # Telegram entegrasyonu
        text = assistant.ask_telegram(question)
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or str(ROOT / "data" / "betting.duckdb")
        self._template_gen = TemplateSQLGenerator()
        self._gemini_gen = GeminiSQLGenerator()
        self._query_history: list[QueryResult] = []
        logger.debug("[Query] Assistant başlatıldı.")

    def ask(self, question: str) -> QueryResult:
        """Doğal dilde soru sor, SQL ile yanıtla."""
        result = QueryResult(question=question)
        start = time.perf_counter()

        # 1) Gemini ile dene
        sql = self._gemini_gen.generate(question)
        if sql:
            result.method = "gemini"
            result.confidence = 0.85
        else:
            # 2) Şablon ile dene
            sql = self._template_gen.generate(question)
            if sql:
                result.method = "template"
                result.confidence = 0.60
            else:
                result.method = "error"
                result.error = "Soru anlaşılamadı. Lütfen farklı ifade edin."
                result.formatted_answer = result.error
                return result

        result.sql = sql

        # SQL çalıştır
        try:
            raw = self._execute_sql(sql)
            result.raw_result = raw
            result.formatted_answer = self._format_result(question, raw, sql)
        except Exception as e:
            result.error = str(e)
            result.formatted_answer = f"Sorgu hatası: {e}"
            result.method = "error"

        elapsed = (time.perf_counter() - start) * 1000
        result.execution_time_ms = round(elapsed, 1)

        self._query_history.append(result)
        return result

    def _execute_sql(self, sql: str) -> Any:
        """SQL sorgusunu çalıştır."""
        if not DUCK_OK:
            return "DuckDB yüklü değil."

        try:
            conn = duckdb.connect(self._db_path, read_only=True)
            result = conn.execute(sql).fetchall()
            columns = [desc[0] for desc in conn.description or []]
            conn.close()
            return {"columns": columns, "rows": result}
        except duckdb.CatalogException:
            # Tablo yoksa demo veri döndür
            return {"columns": ["info"], "rows": [("Tablo bulunamadı – demo mod",)]}
        except Exception as e:
            raise RuntimeError(f"SQL hatası: {e}")

    def _format_result(self, question: str, raw: Any,
                        sql: str) -> str:
        """Sonucu okunabilir formata çevir."""
        if not isinstance(raw, dict):
            return str(raw)

        rows = raw.get("rows", [])
        columns = raw.get("columns", [])

        if not rows:
            return "Sonuç bulunamadı."

        # Tek değer
        if len(rows) == 1 and len(rows[0]) == 1:
            val = rows[0][0]
            if isinstance(val, float):
                return f"Sonuç: %{val:.1f}"
            return f"Sonuç: {val}"

        # Tablo formatı
        lines = []
        if columns:
            header = " | ".join(str(c) for c in columns)
            lines.append(header)
            lines.append("─" * len(header))

        for row in rows[:20]:
            line = " | ".join(
                f"{v:.2f}" if isinstance(v, float) else str(v)
                for v in row
            )
            lines.append(line)

        if len(rows) > 20:
            lines.append(f"... ve {len(rows) - 20} satır daha")

        return "\n".join(lines)

    def ask_telegram(self, question: str) -> str:
        """Telegram için HTML formatında yanıt."""
        result = self.ask(question)

        text = (
            f"🔍 <b>Soru:</b> {result.question}\n\n"
        )

        if result.error:
            text += f"❌ <b>Hata:</b> {result.error}"
        else:
            text += (
                f"📊 <b>Sonuç:</b>\n"
                f"<pre>{result.formatted_answer[:1500]}</pre>\n\n"
                f"🔧 <b>SQL:</b>\n"
                f"<code>{result.sql[:500]}</code>\n\n"
                f"⏱ {result.execution_time_ms:.0f}ms "
                f"({result.method})"
            )

        return text

    @property
    def history(self) -> list[QueryResult]:
        return list(self._query_history)
