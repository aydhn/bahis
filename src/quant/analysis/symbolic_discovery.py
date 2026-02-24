"""
symbolic_discovery.py – Symbolic Regression (PySR).

"Kara kutudan" çıkıyoruz. Verilerin içindeki gerçek matematiksel
denklemi bulur. Bot artık şunu der:
  "Bu ligdeki gol sayısı: Gol = √(xG) × Form^1.2"

Kavramlar:
  - Symbolic Regression: Veri kümesinden analitik formül keşfetme
  - PySR: Julia tabanlı yüksek performanslı evrimsel sembolik regresyon
  - Pareto Front: Karmaşıklık vs doğruluk tradeoff'u — en basit
    ve en doğru formülleri bulur
  - Operatörler: +, -, ×, ÷, sin, cos, exp, log, sqrt, pow
  - Complexity: Formüldeki terim sayısı (daha az = daha iyi)
  - Fitness (Loss): MSE veya MAE — tahmin hatası

Akış:
  1. Maç verilerini al (xG, şut, pas, form, hava, vb.)
  2. PySR evrimsel aramayı başlat (1000+ nesil × 100 birey)
  3. Pareto front'taki formülleri sırala (basitlik vs doğruluk)
  4. En iyi formülü insan okunur LaTeX'e çevir
  5. XAI modülüne gönder → Telegram'da "Maçın Matematiksel Formülü"

Teknoloji: PySR (Julia backend)
Fallback: Genetik programlama ile basit formül arama (deap veya özel)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

try:
    from pysr import PySRRegressor
    PYSR_OK = True
except ImportError:
    PYSR_OK = False
    logger.debug("pysr yüklü değil – genetik formül arama fallback.")

try:
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

ROOT = Path(__file__).resolve().parent.parent.parent
FORMULA_DIR = ROOT / "data" / "formulas"
FORMULA_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
@dataclass
class DiscoveredFormula:
    """Keşfedilen matematiksel formül."""
    equation: str = ""           # İnsan okunur formül
    latex: str = ""              # LaTeX formatı
    complexity: int = 0          # Terim sayısı
    loss: float = 0.0            # Tahmin hatası (MSE)
    r2: float = 0.0              # R² skoru
    score: float = 0.0           # Pareto skoru (doğruluk / karmaşıklık)


@dataclass
class SymbolicReport:
    """Sembolik regresyon raporu."""
    target: str = ""             # Hedef değişken adı
    # Keşifler
    best_formula: DiscoveredFormula = field(default_factory=DiscoveredFormula)
    pareto_front: list[DiscoveredFormula] = field(default_factory=list)
    # Meta
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    n_samples: int = 0
    n_generations: int = 0
    search_time_sec: float = 0.0
    method: str = ""
    recommendation: str = ""


# ═══════════════════════════════════════════════
#  GENETIK FORMÜL ARAMA (Fallback)
# ═══════════════════════════════════════════════
class SimpleSymbolicSearch:
    """Basit genetik formül arama (PySR yoksa).

    Önceden tanımlı formül şablonlarını dener ve en iyisini seçer.
    """

    TEMPLATES = [
        ("a*x0 + b*x1 + c",
         lambda X, p: p[0] * X[:, 0] + p[1] * X[:, 1] + p[2]),
        ("a*sqrt(abs(x0)) + b*x1",
         lambda X, p: p[0] * np.sqrt(np.abs(X[:, 0]) + 1e-6) + p[1] * X[:, 1]),
        ("a*x0^b + c",
         lambda X, p: p[0] * np.power(np.abs(X[:, 0]) + 1e-6, np.clip(p[1], 0.1, 3)) + p[2]),
        ("a*x0*x1 + b*x2 + c",
         lambda X, p: p[0] * X[:, 0] * X[:, min(1, X.shape[1] - 1)] + p[1] * X[:, min(2, X.shape[1] - 1)] + p[2]),
        ("a*log(abs(x0)+1) + b*x1^2 + c",
         lambda X, p: p[0] * np.log(np.abs(X[:, 0]) + 1) + p[1] * X[:, 1] ** 2 + p[2]),
        ("a*exp(-b*x0) + c*x1",
         lambda X, p: p[0] * np.exp(-np.clip(p[1], -2, 2) * X[:, 0]) + p[2] * X[:, 1]),
        ("a*x0/(1 + b*abs(x1)) + c",
         lambda X, p: p[0] * X[:, 0] / (1 + np.abs(p[1]) * np.abs(X[:, 1]) + 1e-6) + p[2]),
    ]

    def search(self, X: np.ndarray, y: np.ndarray,
                 feature_names: list[str] | None = None,
                 n_trials: int = 500) -> list[DiscoveredFormula]:
        """Formül arama."""
        results = []
        n_features = X.shape[1]
        names = feature_names or [f"x{i}" for i in range(n_features)]

        for template_str, func in self.TEMPLATES:
            if n_features < 2 and "x1" in template_str:
                continue

            best_loss = float("inf")
            best_params = [1.0, 1.0, 0.0]

            for _ in range(n_trials):
                params = np.random.randn(3) * 2
                try:
                    y_pred = func(X, params)
                    if not np.all(np.isfinite(y_pred)):
                        continue
                    loss = float(np.mean((y - y_pred) ** 2))
                    if loss < best_loss:
                        best_loss = loss
                        best_params = params.tolist()
                except Exception:
                    continue

            if best_loss < float("inf"):
                # Formül string oluştur
                eq = template_str
                for i, p in enumerate(best_params):
                    eq = eq.replace(chr(ord("a") + i), f"{p:.3f}", 1)
                for i, name in enumerate(names):
                    eq = eq.replace(f"x{i}", name)

                r2 = 0.0
                if SKLEARN_OK:
                    try:
                        y_pred = func(X, best_params)
                        r2 = float(r2_score(y, y_pred))
                    except Exception:
                        pass

                complexity = len(template_str.split("+")) + len(template_str.split("*"))

                results.append(DiscoveredFormula(
                    equation=eq,
                    latex=eq.replace("*", r" \times ").replace("sqrt", r"\sqrt"),
                    complexity=complexity,
                    loss=round(best_loss, 6),
                    r2=round(r2, 4),
                    score=round(r2 / max(complexity, 1), 4),
                ))

        results.sort(key=lambda f: -f.score)
        return results


# ═══════════════════════════════════════════════
#  SYMBOLIC DISCOVERY ENGINE (Ana Sınıf)
# ═══════════════════════════════════════════════
class SymbolicDiscovery:
    """Sembolik regresyon ile formül keşfi.

    Kullanım:
        sd = SymbolicDiscovery()

        # Keşif
        report = sd.discover(X, y, feature_names=["xG", "Şut", "Form"],
                              target="goals")

        # En iyi formül
        print(report.best_formula.equation)
        # "1.23 * sqrt(xG) + 0.45 * Form - 0.12"

        # Tahmin
        y_pred = sd.predict(X_new)
    """

    def __init__(self, max_complexity: int = 20,
                 n_populations: int = 30,
                 population_size: int = 50,
                 n_iterations: int = 40):
        self._max_complexity = max_complexity
        self._n_pops = n_populations
        self._pop_size = population_size
        self._n_iters = n_iterations
        self._model: Any = None
        self._fallback = SimpleSymbolicSearch()
        self._best_formula: DiscoveredFormula | None = None

        logger.debug(
            f"[Symbolic] Engine başlatıldı: max_cplx={max_complexity}, "
            f"pysr={'OK' if PYSR_OK else 'fallback'}"
        )

    def discover(self, X: np.ndarray, y: np.ndarray,
                   feature_names: list[str] | None = None,
                   target: str = "goals") -> SymbolicReport:
        """Sembolik regresyon ile formül keşfet."""
        report = SymbolicReport(target=target)
        t0 = time.perf_counter()

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        report.n_samples = len(X)
        report.n_features = X.shape[1]
        report.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]

        if len(X) < 20:
            report.recommendation = "Yetersiz veri (min 20)."
            report.method = "none"
            return report

        if PYSR_OK:
            report = self._discover_pysr(X, y, report)
        else:
            report = self._discover_fallback(X, y, report)

        report.search_time_sec = round(time.perf_counter() - t0, 2)
        report.recommendation = self._advice(report)

        # Kaydet
        if report.best_formula.equation:
            self._best_formula = report.best_formula
            self._save_formula(report)

        return report

    def _discover_pysr(self, X: np.ndarray, y: np.ndarray,
                         report: SymbolicReport) -> SymbolicReport:
        """PySR ile sembolik regresyon."""
        try:
            model = PySRRegressor(
                niterations=self._n_iters,
                populations=self._n_pops,
                population_size=self._pop_size,
                maxsize=self._max_complexity,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log", "exp", "abs"],
                loss="loss(prediction, target) = (prediction - target)^2",
                temp_equation_file=True,
                verbosity=0,
                progress=False,
            )

            model.fit(X, y, variable_names=report.feature_names)
            self._model = model

            # Pareto front
            for i, eq in enumerate(model.equations_):
                try:
                    formula = DiscoveredFormula(
                        equation=str(model.sympy(i)),
                        latex=str(model.latex(i)) if hasattr(model, "latex") else "",
                        complexity=int(eq.get("complexity", i + 1)) if isinstance(eq, dict) else i + 1,
                        loss=float(eq.get("loss", 0)) if isinstance(eq, dict) else 0,
                    )
                    if SKLEARN_OK:
                        y_pred = model.predict(X, index=i)
                        formula.r2 = round(float(r2_score(y, y_pred)), 4)
                    formula.score = round(
                        formula.r2 / max(formula.complexity, 1), 4,
                    )
                    report.pareto_front.append(formula)
                except Exception:
                    continue

            if report.pareto_front:
                report.best_formula = max(report.pareto_front, key=lambda f: f.score)
            report.n_generations = self._n_iters
            report.method = "pysr_evolutionary"

        except Exception as e:
            logger.warning(f"[Symbolic] PySR hatası: {e}")
            report = self._discover_fallback(X, y, report)

        return report

    def _discover_fallback(self, X: np.ndarray, y: np.ndarray,
                              report: SymbolicReport) -> SymbolicReport:
        """Fallback formül arama."""
        results = self._fallback.search(
            X, y, feature_names=report.feature_names,
        )

        report.pareto_front = results[:10]
        if results:
            report.best_formula = results[0]
        report.method = "template_search"
        report.n_generations = 500
        return report

    def predict(self, X: np.ndarray, index: int = -1) -> np.ndarray:
        """Keşfedilen formül ile tahmin."""
        X = np.array(X, dtype=np.float64)

        if PYSR_OK and self._model is not None:
            try:
                if index >= 0:
                    return self._model.predict(X, index=index)
                return self._model.predict(X)
            except Exception:
                pass

        return np.zeros(len(X))

    def get_formula_text(self) -> str:
        """En iyi formülü metin olarak döndür."""
        if self._best_formula:
            return self._best_formula.equation
        return "Formül henüz keşfedilmedi."

    def _save_formula(self, report: SymbolicReport) -> None:
        """Formülü dosyaya kaydet."""
        path = FORMULA_DIR / f"formula_{report.target}.txt"
        lines = [
            f"Hedef: {report.target}",
            f"Metod: {report.method}",
            f"En İyi: {report.best_formula.equation}",
            f"R²: {report.best_formula.r2}",
            f"Karmaşıklık: {report.best_formula.complexity}",
            f"Süre: {report.search_time_sec}s",
            "",
            "Pareto Front:",
        ]
        for i, f in enumerate(report.pareto_front[:10]):
            lines.append(f"  {i + 1}. {f.equation} (R²={f.r2}, cplx={f.complexity})")

        path.write_text("\n".join(lines), encoding="utf-8")

    def _advice(self, r: SymbolicReport) -> str:
        if not r.best_formula.equation:
            return "Formül keşfedilemedi. Daha fazla veri gerekli."
        if r.best_formula.r2 > 0.7:
            return (
                f"MÜKEMMEL: {r.target} = {r.best_formula.equation} "
                f"(R²={r.best_formula.r2:.1%}, cplx={r.best_formula.complexity}). "
                f"Bu formül gerçek ilişkiyi yakalıyor."
            )
        if r.best_formula.r2 > 0.4:
            return (
                f"İYİ: {r.target} ≈ {r.best_formula.equation} "
                f"(R²={r.best_formula.r2:.1%}). "
                f"Kısmen açıklayıcı, daha fazla özellik gerekebilir."
            )
        return (
            f"ZAYIF: R²={r.best_formula.r2:.1%}. "
            f"İlişki karmaşık veya veri yetersiz."
        )
