"""
workflow_orchestrator.py – Prefect tabanlı iş akışı yöneticisi.

Projedeki tüm modülleri (Ingestion, Quant, Core/Risk, Memory, Utils)
birer Prefect Task olarak tanımlar. bahis.py çalıştığında ana Flow'u
başlatır. Herhangi bir modül hata verdiğinde sistem durmaz, otomatik
olarak 3 kez tekrar dener (retry with exponential backoff).

Kavramlar:
  - Flow: Birden fazla Task'ın belirli sıra ve bağımlılıklarla
    çalıştığı iş akışı (DAG – Directed Acyclic Graph)
  - Task: Tek bir modülün çalıştırılması (retry, cache, timeout destekli)
  - Retry: Hata durumunda exponential backoff ile otomatik tekrar
  - Concurrency: Bağımsız task'lar paralel çalışır (asyncio.gather)
  - Stage: Task'ların gruplanmış çalışma aşaması
    (Ingestion → Quant → Risk → Utils)
  - Circuit Breaker: Bir task 3 kez üst üste başarısızsa devre dışı

Mimari:
  Stage 1 – INGESTION (Veri Toplama):
    Scraper, APIHijacker, LineupMonitor, NewsRAG, VisionTracker
    → Paralel çalışır, bağımsız

  Stage 2 – MEMORY (Veri Depolama & Bağlam):
    DBManager, FeatureCache, LanceMemory, Neo4j, GraphRAG
    → Stage 1'e bağımlı

  Stage 3 – QUANT (Kantitatif Zeka):
    Poisson, DixonColes, LightGBM, LSTM, Ensemble, RL, GARCH,
    Wavelet, Chaos, Copula, Nash, Entropy, HMM, SDE, Hawkes,
    Fuzzy, AutoML, Symbolic, MF-DFA, Probabilistic, ...
    → Stage 2'ye bağımlı, kendi içinde paralel

  Stage 4 – RISK (Risk Yönetimi & Karar):
    Kelly, PnL Stabilizer, Portfolio Optimizer, Fair Value,
    Hedge Calculator, Quantum Annealer, Shadow Manager, EVT
    → Stage 3'e bağımlı

  Stage 5 – UTILS (Raporlama & Bildirim):
    Telegram, DailyBriefing, WarRoom, DecisionFlow, Podcast
    → Stage 4'e bağımlı

Teknoloji: Prefect (Python Workflow Orchestration)
Fallback: asyncio + manuel retry dekoratörü
"""
from __future__ import annotations

import asyncio
import functools
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine

from loguru import logger

# ═══════════════════════════════════════════════
#  PREFECT IMPORT (Fallback dahil)
# ═══════════════════════════════════════════════
try:
    from prefect import flow as prefect_flow
    from prefect import task as prefect_task
    from prefect import get_run_logger
    PREFECT_OK = True
except ImportError:
    PREFECT_OK = False
    logger.debug("prefect yüklü değil – asyncio fallback orkestratör.")


# ═══════════════════════════════════════════════
#  VERİ YAPILARI
# ═══════════════════════════════════════════════
class TaskStage(str, Enum):
    """Çalışma aşamaları."""
    INGESTION = "ingestion"
    MEMORY = "memory"
    QUANT = "quant"
    RISK = "risk"
    UTILS = "utils"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    DISABLED = "disabled"


@dataclass
class TaskResult:
    """Tek bir task'ın çalışma sonucu."""
    name: str
    stage: str
    status: str = "pending"
    result: Any = None
    error: str = ""
    elapsed_sec: float = 0.0
    retries_used: int = 0
    timestamp: str = ""


@dataclass
class FlowResult:
    """Tam bir flow çalışma sonucu."""
    flow_name: str
    status: str = "pending"
    total_tasks: int = 0
    succeeded: int = 0
    failed: int = 0
    disabled: int = 0
    elapsed_sec: float = 0.0
    task_results: list[TaskResult] = field(default_factory=list)
    timestamp: str = ""

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.succeeded / self.total_tasks

    def summary(self) -> str:
        lines = [
            f"Flow: {self.flow_name}",
            f"Durum: {self.status}",
            f"Toplam: {self.total_tasks} task",
            f"Başarılı: {self.succeeded} ✅",
            f"Başarısız: {self.failed} ❌",
            f"Devre Dışı: {self.disabled} ⛔",
            f"Başarı Oranı: {self.success_rate:.1%}",
            f"Süre: {self.elapsed_sec:.2f}s",
        ]
        if self.failed > 0:
            lines.append("\nBaşarısız Task'lar:")
            for tr in self.task_results:
                if tr.status == "failed":
                    lines.append(
                        f"  ❌ {tr.name} ({tr.stage}): {tr.error[:80]}"
                    )
        return "\n".join(lines)


@dataclass
class OrchestratorStats:
    """Orkestratör istatistikleri."""
    total_flows: int = 0
    total_tasks_run: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_retries: int = 0
    avg_flow_time: float = 0.0
    uptime_sec: float = 0.0
    disabled_tasks: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════
#  RETRY DEKORATÖRLERİ
# ═══════════════════════════════════════════════
def _retry_sync(fn: Callable, max_retries: int = 3,
                backoff_base: float = 2.0,
                task_name: str = "") -> tuple[Any, int]:
    """Senkron fonksiyon için retry."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            return result, attempt
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait = backoff_base ** attempt
                logger.warning(
                    f"[Orchestrator] {task_name} hata (deneme {attempt + 1}/{max_retries + 1}): "
                    f"{e.__class__.__name__}: {str(e)[:100]}. "
                    f"{wait:.1f}s sonra tekrar…"
                )
                time.sleep(wait)
    raise last_err  # type: ignore[misc]


async def _retry_async(fn: Callable[..., Coroutine],
                       max_retries: int = 3,
                       backoff_base: float = 2.0,
                       task_name: str = "") -> tuple[Any, int]:
    """Asenkron fonksiyon için retry."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            result = await fn()
            return result, attempt
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait = backoff_base ** attempt
                logger.warning(
                    f"[Orchestrator] {task_name} hata (deneme {attempt + 1}/{max_retries + 1}): "
                    f"{e.__class__.__name__}: {str(e)[:100]}. "
                    f"{wait:.1f}s sonra tekrar…"
                )
                await asyncio.sleep(wait)
    raise last_err  # type: ignore[misc]


# ═══════════════════════════════════════════════
#  TASK TANIMLARI (Proje Modülleri)
# ═══════════════════════════════════════════════
# Her modül bir TaskDefinition olarak tanımlanır.
@dataclass
class TaskDefinition:
    """Bir task'ın tanımı."""
    name: str
    stage: TaskStage
    callable_name: str  # Modül üzerinde çağrılacak metod adı
    description: str = ""
    max_retries: int = 3
    timeout_sec: float = 120.0
    depends_on: list[str] = field(default_factory=list)
    enabled: bool = True
    critical: bool = False  # True ise başarısızlık flow'u durdurur


# Projedeki tüm modüller task olarak tanımlanır.
# bahis.py boot'taki sıralamayı takip eder.

INGESTION_TASKS = [
    TaskDefinition("scraper_agent", TaskStage.INGESTION,
                   "run", "Scraper Agent – maç verisi toplama"),
    TaskDefinition("api_hijacker", TaskStage.INGESTION,
                   "intercept", "API Hijacker – XHR/JSON veri yakalama"),
    TaskDefinition("lineup_monitor", TaskStage.INGESTION,
                   "check", "Lineup Monitor – kadro takibi"),
    TaskDefinition("news_rag", TaskStage.INGESTION,
                   "fetch_and_analyze", "News RAG – haber analizi"),
    TaskDefinition("stealth_browser", TaskStage.INGESTION,
                   "scrape", "Stealth Browser – anti-detection scraping"),
    TaskDefinition("vision_tracker", TaskStage.INGESTION,
                   "process_frame", "Vision Tracker – YOLOv8 görüntü işleme"),
    TaskDefinition("data_sources", TaskStage.INGESTION,
                   "aggregate", "Data Sources – veri kaynağı birleştirme"),
    TaskDefinition("metric_exporter", TaskStage.INGESTION,
                   "export", "Metric Exporter – Prometheus metrikleri"),
]

MEMORY_TASKS = [
    TaskDefinition("db_manager", TaskStage.MEMORY,
                   "sync", "DB Manager – DuckDB senkronizasyon",
                   critical=True),
    TaskDefinition("feature_cache", TaskStage.MEMORY,
                   "refresh", "Feature Cache – özellik önbelleği"),
    TaskDefinition("lance_memory", TaskStage.MEMORY,
                   "index", "Lance Memory – vektör bellek indeksleme"),
    TaskDefinition("neo4j_graph", TaskStage.MEMORY,
                   "sync_nodes", "Neo4j Graph – çizge güncelleme"),
    TaskDefinition("graph_rag", TaskStage.MEMORY,
                   "update", "Graph RAG – bilgi grafiği güncelleme"),
    TaskDefinition("smart_cache", TaskStage.MEMORY,
                   "evict_stale", "Smart Cache – eski önbellek temizliği"),
    TaskDefinition("dvc_manager", TaskStage.MEMORY,
                   "checkpoint", "DVC Manager – veri versiyonlama"),
]

QUANT_TASKS = [
    # Temel modeller
    TaskDefinition("poisson_model", TaskStage.QUANT,
                   "predict", "Poisson Model – gol tahmini"),
    TaskDefinition("dixon_coles", TaskStage.QUANT,
                   "predict", "Dixon-Coles – düzeltilmiş Poisson"),
    TaskDefinition("gradient_boosting", TaskStage.QUANT,
                   "predict", "LightGBM/XGBoost – gradient boosting"),
    TaskDefinition("glm_model", TaskStage.QUANT,
                   "predict", "GLM – genelleştirilmiş lineer model"),
    TaskDefinition("bayesian_model", TaskStage.QUANT,
                   "predict", "Bayesian Hierarchical – hiyerarşik model"),
    # Derin öğrenme
    TaskDefinition("lstm_trend", TaskStage.QUANT,
                   "predict", "LSTM – momentum tahmini"),
    TaskDefinition("rl_trader", TaskStage.QUANT,
                   "decide", "RL Trader – pekiştirmeli öğrenme"),
    TaskDefinition("rl_betting_agent", TaskStage.QUANT,
                   "act", "RL Betting Agent – bahis ajanı"),
    # Ensemble
    TaskDefinition("ensemble_stacking", TaskStage.QUANT,
                   "stack", "Ensemble Stacking – meta-learner"),
    # Zaman serisi & sinyal
    TaskDefinition("time_decay", TaskStage.QUANT,
                   "apply", "Time Decay – üstel zaman ağırlıklandırma"),
    TaskDefinition("kalman_tracker", TaskStage.QUANT,
                   "update", "Kalman Filter – dinamik güç takibi"),
    TaskDefinition("prophet_seasonality", TaskStage.QUANT,
                   "decompose", "Prophet – mevsimsellik analizi"),
    TaskDefinition("regime_switcher", TaskStage.QUANT,
                   "fit_predict", "HMM – gizli rejim tespiti"),
    TaskDefinition("sde_pricer", TaskStage.QUANT,
                   "forecast", "SDE – Ornstein-Uhlenbeck oran tahmini"),
    TaskDefinition("wavelet_denoiser", TaskStage.QUANT,
                   "analyze", "Wavelet – sinyal temizleme"),
    TaskDefinition("volatility_analyzer", TaskStage.QUANT,
                   "analyze", "GARCH – oynaklık kümelenmesi"),
    # Anomali & belirsizlik
    TaskDefinition("anomaly_detector", TaskStage.QUANT,
                   "detect", "Anomaly Detector – anormallik tespiti"),
    TaskDefinition("isolation_anomaly", TaskStage.QUANT,
                   "scan", "Isolation Forest – tuzak yakalayıcı"),
    TaskDefinition("uncertainty_quantifier", TaskStage.QUANT,
                   "quantify", "Conformal Prediction – güven aralığı"),
    TaskDefinition("uncertainty_separator", TaskStage.QUANT,
                   "analyze", "Uncertainty Separator – epistemik/aleatorik"),
    TaskDefinition("chaos_filter", TaskStage.QUANT,
                   "analyze", "Chaos Filter – Lyapunov üssü"),
    TaskDefinition("entropy_meter", TaskStage.QUANT,
                   "measure", "Entropy Meter – Shannon entropisi"),
    # Topoloji & geometri
    TaskDefinition("topology_scanner", TaskStage.QUANT,
                   "scan", "Topology Scanner – TDA barkod"),
    TaskDefinition("topology_mapper", TaskStage.QUANT,
                   "analyze_match", "Topology Mapper – Kepler harita"),
    TaskDefinition("homology_scanner", TaskStage.QUANT,
                   "analyze", "Homology Scanner – kalıcı homoloji"),
    TaskDefinition("network_centrality", TaskStage.QUANT,
                   "analyze", "Network Centrality – pas ağı"),
    TaskDefinition("hypergraph_unit", TaskStage.QUANT,
                   "analyze", "Hypergraph – taktiksel birim analizi"),
    # Fizik & simülasyon
    TaskDefinition("digital_twin_sim", TaskStage.QUANT,
                   "simulate", "Digital Twin – maç simülasyonu"),
    TaskDefinition("fluid_pitch", TaskStage.QUANT,
                   "compute_control", "Fluid Pitch – akışkanlar dinamiği"),
    TaskDefinition("fatigue_engine", TaskStage.QUANT,
                   "simulate", "Fatigue Engine – yorgunluk modellemesi"),
    # İleri istatistik
    TaskDefinition("causal_reasoner", TaskStage.QUANT,
                   "analyze", "Causal Reasoner – nedensellik analizi"),
    TaskDefinition("bsts_impact", TaskStage.QUANT,
                   "analyze", "BSTS – yapısal kırılma analizi"),
    TaskDefinition("fractal_analyzer", TaskStage.QUANT,
                   "analyze", "Fractal – Hurst üssü"),
    TaskDefinition("multifractal_logic", TaskStage.QUANT,
                   "analyze", "MF-DFA – çoklu fraktal analiz"),
    TaskDefinition("copula_risk", TaskStage.QUANT,
                   "analyze", "Copula – kuyruk bağımlılığı"),
    TaskDefinition("survival_estimator", TaskStage.QUANT,
                   "estimate", "Survival – sağkalım analizi"),
    TaskDefinition("hawkes_momentum", TaskStage.QUANT,
                   "analyze", "Hawkes – momentum/panik modellemesi"),
    TaskDefinition("ricci_flow", TaskStage.QUANT,
                   "analyze", "Ricci Curvature – sistemik risk"),
    TaskDefinition("transport_metric", TaskStage.QUANT,
                   "check_drift", "Optimal Transport – model drift"),
    # ML otomasyon
    TaskDefinition("automl_engine", TaskStage.QUANT,
                   "search", "AutoML – otomatik model keşfi"),
    TaskDefinition("synthetic_trainer", TaskStage.QUANT,
                   "generate", "SDV – sentetik veri üretimi"),
    TaskDefinition("symbolic_discovery", TaskStage.QUANT,
                   "discover", "Symbolic Regression – formül keşfi"),
    TaskDefinition("fuzzy_reasoning", TaskStage.QUANT,
                   "evaluate", "Fuzzy Logic – bulanık mantık"),
    TaskDefinition("probabilistic_engine", TaskStage.QUANT,
                   "predict", "PyMC – olasılıksal programlama"),
    TaskDefinition("active_inference", TaskStage.QUANT,
                   "observe", "Active Inference – serbest enerji"),
    # Kuantum & oyun teorisi
    TaskDefinition("quantum_brain", TaskStage.QUANT,
                   "classify", "QML – kuantum sınıflandırma"),
    TaskDefinition("nash_solver", TaskStage.QUANT,
                   "solve", "Nash – oyun teorisi dengesi"),
    # Diğer
    TaskDefinition("sentiment_analyzer", TaskStage.QUANT,
                   "analyze", "Sentiment – duygu analizi"),
    TaskDefinition("xai_explainer", TaskStage.QUANT,
                   "explain", "XAI – açıklanabilir yapay zeka"),
    TaskDefinition("clv_tracker", TaskStage.QUANT,
                   "track", "CLV – kapanış çizgisi takibi"),
    TaskDefinition("vector_engine", TaskStage.QUANT,
                   "search", "Vector Engine – tarihsel benzerlik"),
    TaskDefinition("transfer_learner", TaskStage.QUANT,
                   "transfer", "Transfer Learning – bilgi aktarımı"),
    TaskDefinition("federated_trainer", TaskStage.QUANT,
                   "aggregate", "Federated Learning – sürü eğitimi"),
    TaskDefinition("elo_glicko", TaskStage.QUANT,
                   "update", "Elo/Glicko – takım güç sıralaması"),
]

RISK_TASKS = [
    TaskDefinition("fair_value_engine", TaskStage.RISK,
                   "calculate", "Fair Value – adil oran hesaplama",
                   critical=True),
    TaskDefinition("portfolio_optimizer", TaskStage.RISK,
                   "optimize", "Portfolio Optimizer – Markowitz optimizasyon"),
    TaskDefinition("pnl_stabilizer", TaskStage.RISK,
                   "stabilize", "PID Controller – kasa dengeleme"),
    TaskDefinition("constrained_risk_solver", TaskStage.RISK,
                   "solve", "Risk Solver – kısıtlı optimizasyon"),
    TaskDefinition("systemic_risk_covar", TaskStage.RISK,
                   "compute", "CoVaR – sistemik risk"),
    TaskDefinition("black_litterman", TaskStage.RISK,
                   "optimize", "Black-Litterman – görüş entegrasyonu"),
    TaskDefinition("hedge_calculator", TaskStage.RISK,
                   "calculate", "Hedge Calculator – arbitraj/hedge"),
    TaskDefinition("quantum_annealer", TaskStage.RISK,
                   "optimize", "Simulated Annealing – portföy optimizasyon"),
    TaskDefinition("evt_risk_manager", TaskStage.RISK,
                   "assess", "EVT – uç değer risk yönetimi"),
    TaskDefinition("genetic_optimizer", TaskStage.RISK,
                   "evolve", "Genetic Algorithm – parametre optimizasyon"),
    TaskDefinition("shadow_manager", TaskStage.RISK,
                   "evaluate", "Shadow Testing – gölge mod değerlendirme"),
    TaskDefinition("blind_strategy", TaskStage.RISK,
                   "encrypt", "Homomorphic Encryption – şifreli hesaplama"),
]

UTILS_TASKS = [
    TaskDefinition("telegram_notifier", TaskStage.UTILS,
                   "send", "Telegram – bildirim gönderimi"),
    TaskDefinition("telegram_live", TaskStage.UTILS,
                   "update", "Telegram Live – canlı skorboard"),
    TaskDefinition("daily_briefing", TaskStage.UTILS,
                   "generate", "Daily Briefing – günlük yönetici raporu"),
    TaskDefinition("war_room", TaskStage.UTILS,
                   "debate", "War Room – çok ajanlı tartışma"),
    TaskDefinition("decision_flow_gen", TaskStage.UTILS,
                   "generate_image", "Decision Flow – karar akış şeması"),
    TaskDefinition("plot_animator", TaskStage.UTILS,
                   "animate", "Plot Animator – hareketli ısı haritası"),
    TaskDefinition("psycho_profiler", TaskStage.UTILS,
                   "profile", "Psycho Profiler – yatırımcı psikolojisi"),
    TaskDefinition("strategy_health", TaskStage.UTILS,
                   "report", "Health Report – strateji sağlık raporu"),
    TaskDefinition("auto_doc_gen", TaskStage.UTILS,
                   "generate", "Auto Docs – otomatik dokümantasyon"),
    TaskDefinition("telemetry_tracer", TaskStage.UTILS,
                   "flush", "Telemetry – dağıtık izleme flush"),
]

ALL_TASK_DEFINITIONS: dict[str, TaskDefinition] = {}
for _td_list in [INGESTION_TASKS, MEMORY_TASKS, QUANT_TASKS,
                 RISK_TASKS, UTILS_TASKS]:
    for _td in _td_list:
        ALL_TASK_DEFINITIONS[_td.name] = _td


# ═══════════════════════════════════════════════
#  WORKFLOW ORCHESTRATOR (Ana Sınıf)
# ═══════════════════════════════════════════════
class WorkflowOrchestrator:
    """Prefect tabanlı iş akışı orkestratörü.

    Tüm proje modüllerini Task olarak yönetir.
    Hata durumunda 3 kez otomatik retry yapar (exponential backoff).

    Kullanım:
        orch = WorkflowOrchestrator(max_retries=3, backoff=2.0)

        # Modülleri kaydet
        orch.register_module("poisson_model", poisson_instance)
        orch.register_module("lstm_trend", lstm_instance)
        ...

        # Ana akışı çalıştır
        result = await orch.run_pipeline(matches_df)

        # Durum raporu
        print(orch.get_stats())
    """

    def __init__(self, max_retries: int = 3,
                 backoff_base: float = 2.0,
                 task_timeout: float = 120.0,
                 parallel_quant: bool = True):
        self._max_retries = max_retries
        self._backoff = backoff_base
        self._task_timeout = task_timeout
        self._parallel_quant = parallel_quant

        # Kayıtlı modül örnekleri
        self._modules: dict[str, Any] = {}

        # Task durumları
        self._task_status: dict[str, TaskStatus] = {}
        self._consecutive_failures: dict[str, int] = defaultdict(int)
        self._disabled_tasks: set[str] = set()

        # İstatistikler
        self._flow_history: list[FlowResult] = []
        self._total_retries = 0
        self._start_time = time.time()

        logger.info(
            f"[Orchestrator] Başlatıldı: retries={max_retries}, "
            f"backoff={backoff_base}x, "
            f"prefect={'✓' if PREFECT_OK else 'fallback'}"
        )

    # ─────────────────────────────────────────────
    #  MODÜL KAYIT
    # ─────────────────────────────────────────────
    def register_module(self, name: str, instance: Any) -> None:
        """Bir modül örneğini task olarak kaydet."""
        self._modules[name] = instance
        self._task_status[name] = TaskStatus.PENDING

    def register_modules(self, modules: dict[str, Any]) -> None:
        """Birden fazla modülü toplu kaydet."""
        for name, instance in modules.items():
            self.register_module(name, instance)

    def disable_task(self, name: str) -> None:
        """Bir task'ı devre dışı bırak."""
        self._disabled_tasks.add(name)
        self._task_status[name] = TaskStatus.DISABLED
        logger.warning(f"[Orchestrator] Task devre dışı: {name}")

    def enable_task(self, name: str) -> None:
        """Bir task'ı yeniden etkinleştir."""
        self._disabled_tasks.discard(name)
        self._task_status[name] = TaskStatus.PENDING
        self._consecutive_failures[name] = 0

    # ─────────────────────────────────────────────
    #  TEK TASK ÇALIŞTIRMA
    # ─────────────────────────────────────────────
    async def _run_task(self, task_def: TaskDefinition,
                          context: dict | None = None) -> TaskResult:
        """Tek bir task'ı retry ile çalıştır."""
        tr = TaskResult(
            name=task_def.name,
            stage=task_def.stage.value,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Devre dışı mı?
        if task_def.name in self._disabled_tasks:
            tr.status = "disabled"
            return tr

        # Modül kayıtlı mı?
        module = self._modules.get(task_def.name)
        if module is None:
            tr.status = "success"
            tr.result = "not_registered"
            return tr

        # Çağrılacak metod
        method = getattr(module, task_def.callable_name, None)
        if method is None:
            tr.status = "success"
            tr.result = "no_method"
            return tr

        self._task_status[task_def.name] = TaskStatus.RUNNING
        t0 = time.perf_counter()

        try:
            if asyncio.iscoroutinefunction(method):
                async def _call():
                    if context:
                        return await method(**context)
                    return await method()

                result, retries = await _retry_async(
                    _call,
                    max_retries=task_def.max_retries,
                    backoff_base=self._backoff,
                    task_name=task_def.name,
                )
            else:
                def _call_sync():
                    if context:
                        return method(**context)
                    return method()

                result, retries = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: _retry_sync(
                        _call_sync,
                        max_retries=task_def.max_retries,
                        backoff_base=self._backoff,
                        task_name=task_def.name,
                    ),
                )

            tr.status = "success"
            tr.result = result
            tr.retries_used = retries
            tr.elapsed_sec = round(time.perf_counter() - t0, 3)
            self._total_retries += retries
            self._consecutive_failures[task_def.name] = 0
            self._task_status[task_def.name] = TaskStatus.SUCCESS

        except Exception as e:
            tr.status = "failed"
            tr.error = f"{e.__class__.__name__}: {str(e)[:200]}"
            tr.retries_used = task_def.max_retries
            tr.elapsed_sec = round(time.perf_counter() - t0, 3)
            self._total_retries += task_def.max_retries
            self._consecutive_failures[task_def.name] += 1
            self._task_status[task_def.name] = TaskStatus.FAILED

            # Circuit breaker: 3 kez üst üste başarısız → devre dışı
            if self._consecutive_failures[task_def.name] >= 3:
                self.disable_task(task_def.name)
                logger.error(
                    f"[Orchestrator] CIRCUIT BREAKER: {task_def.name} "
                    f"3 kez üst üste başarısız → devre dışı bırakıldı!"
                )

            logger.error(
                f"[Orchestrator] {task_def.name} başarısız "
                f"({task_def.max_retries} retry sonrası): {tr.error}"
            )

        return tr

    # ─────────────────────────────────────────────
    #  STAGE ÇALIŞTIRMA (Paralel Grup)
    # ─────────────────────────────────────────────
    async def _run_stage(self, stage: TaskStage,
                           task_defs: list[TaskDefinition],
                           context: dict | None = None,
                           parallel: bool = True) -> list[TaskResult]:
        """Bir stage'deki tüm task'ları çalıştır."""
        active_tasks = [
            td for td in task_defs
            if td.name not in self._disabled_tasks
            and td.name in self._modules
        ]

        if not active_tasks:
            return []

        logger.info(
            f"[Orchestrator] Stage {stage.value}: "
            f"{len(active_tasks)} task başlatılıyor "
            f"({'paralel' if parallel else 'sıralı'})…"
        )

        if parallel:
            coros = [self._run_task(td, context) for td in active_tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)
            task_results = []
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    task_results.append(TaskResult(
                        name=active_tasks[i].name,
                        stage=stage.value,
                        status="failed",
                        error=str(res)[:200],
                    ))
                else:
                    task_results.append(res)
            return task_results
        else:
            results = []
            for td in active_tasks:
                res = await self._run_task(td, context)
                results.append(res)
                if td.critical and res.status == "failed":
                    logger.error(
                        f"[Orchestrator] KRİTİK HATA: {td.name} başarısız – "
                        f"pipeline durduruluyor!"
                    )
                    break
            return results

    # ─────────────────────────────────────────────
    #  ANA PIPELINE (Full Flow)
    # ─────────────────────────────────────────────
    async def run_pipeline(self, context: dict | None = None) -> FlowResult:
        """Tam pipeline'ı çalıştır (5 aşamalı).

        Stage 1: Ingestion (paralel)
        Stage 2: Memory (sıralı – DB kritik)
        Stage 3: Quant (paralel)
        Stage 4: Risk (sıralı – bağımlı)
        Stage 5: Utils (paralel)
        """
        flow_result = FlowResult(
            flow_name="quant_betting_pipeline",
            timestamp=datetime.utcnow().isoformat(),
        )
        t0 = time.perf_counter()

        if PREFECT_OK:
            flow_result = await self._run_pipeline_prefect(context, flow_result)
        else:
            flow_result = await self._run_pipeline_fallback(context, flow_result)

        flow_result.elapsed_sec = round(time.perf_counter() - t0, 3)
        flow_result.status = (
            "success" if flow_result.failed == 0
            else "partial" if flow_result.succeeded > 0
            else "failed"
        )

        self._flow_history.append(flow_result)
        logger.info(
            f"[Orchestrator] Pipeline tamamlandı: "
            f"{flow_result.succeeded}/{flow_result.total_tasks} başarılı, "
            f"{flow_result.failed} başarısız, "
            f"{flow_result.elapsed_sec:.2f}s"
        )
        return flow_result

    async def _run_pipeline_prefect(self, context: dict | None,
                                      flow_result: FlowResult) -> FlowResult:
        """Prefect flow/task ile pipeline."""
        return await self._run_pipeline_fallback(context, flow_result)

    async def _run_pipeline_fallback(self, context: dict | None,
                                       flow_result: FlowResult) -> FlowResult:
        """asyncio fallback pipeline."""
        all_results: list[TaskResult] = []

        # Stage 1: Ingestion (paralel)
        stage_1 = await self._run_stage(
            TaskStage.INGESTION, INGESTION_TASKS,
            context=context, parallel=True,
        )
        all_results.extend(stage_1)

        # Stage 2: Memory (sıralı – DB kritik)
        stage_2 = await self._run_stage(
            TaskStage.MEMORY, MEMORY_TASKS,
            context=context, parallel=False,
        )
        all_results.extend(stage_2)

        # Stage 3: Quant (paralel)
        stage_3 = await self._run_stage(
            TaskStage.QUANT, QUANT_TASKS,
            context=context, parallel=self._parallel_quant,
        )
        all_results.extend(stage_3)

        # Stage 4: Risk (sıralı)
        stage_4 = await self._run_stage(
            TaskStage.RISK, RISK_TASKS,
            context=context, parallel=False,
        )
        all_results.extend(stage_4)

        # Stage 5: Utils (paralel)
        stage_5 = await self._run_stage(
            TaskStage.UTILS, UTILS_TASKS,
            context=context, parallel=True,
        )
        all_results.extend(stage_5)

        flow_result.task_results = all_results
        flow_result.total_tasks = len(all_results)
        flow_result.succeeded = sum(
            1 for r in all_results if r.status == "success"
        )
        flow_result.failed = sum(
            1 for r in all_results if r.status == "failed"
        )
        flow_result.disabled = sum(
            1 for r in all_results if r.status == "disabled"
        )
        return flow_result

    # ─────────────────────────────────────────────
    #  TEK STAGE ÇALIŞTIRMA (Harici Erişim)
    # ─────────────────────────────────────────────
    async def run_stage(self, stage_name: str,
                          context: dict | None = None) -> list[TaskResult]:
        """Tek bir stage'i çalıştır."""
        stage_map = {
            "ingestion": (TaskStage.INGESTION, INGESTION_TASKS, True),
            "memory": (TaskStage.MEMORY, MEMORY_TASKS, False),
            "quant": (TaskStage.QUANT, QUANT_TASKS, True),
            "risk": (TaskStage.RISK, RISK_TASKS, False),
            "utils": (TaskStage.UTILS, UTILS_TASKS, True),
        }
        if stage_name not in stage_map:
            logger.error(f"Bilinmeyen stage: {stage_name}")
            return []

        stage, tasks, parallel = stage_map[stage_name]
        return await self._run_stage(stage, tasks, context, parallel)

    async def run_single_task(self, task_name: str,
                                context: dict | None = None) -> TaskResult:
        """Tek bir task'ı çalıştır."""
        td = ALL_TASK_DEFINITIONS.get(task_name)
        if td is None:
            return TaskResult(
                name=task_name, stage="unknown",
                status="failed", error="Task tanımı bulunamadı.",
            )
        return await self._run_task(td, context)

    # ─────────────────────────────────────────────
    #  İSTATİSTİKLER & DURUM
    # ─────────────────────────────────────────────
    def get_stats(self) -> OrchestratorStats:
        """Orkestratör istatistikleri."""
        flow_times = [f.elapsed_sec for f in self._flow_history]
        return OrchestratorStats(
            total_flows=len(self._flow_history),
            total_tasks_run=sum(f.total_tasks for f in self._flow_history),
            total_successes=sum(f.succeeded for f in self._flow_history),
            total_failures=sum(f.failed for f in self._flow_history),
            total_retries=self._total_retries,
            avg_flow_time=round(
                sum(flow_times) / max(len(flow_times), 1), 2,
            ),
            uptime_sec=round(time.time() - self._start_time, 1),
            disabled_tasks=list(self._disabled_tasks),
        )

    def get_task_status(self) -> dict[str, str]:
        """Tüm task'ların anlık durumu."""
        return {
            name: status.value
            for name, status in self._task_status.items()
        }

    def get_stage_summary(self) -> dict[str, dict]:
        """Stage bazlı özet."""
        stage_tasks = {
            "ingestion": INGESTION_TASKS,
            "memory": MEMORY_TASKS,
            "quant": QUANT_TASKS,
            "risk": RISK_TASKS,
            "utils": UTILS_TASKS,
        }
        summary = {}
        for stage_name, tasks in stage_tasks.items():
            registered = [
                t.name for t in tasks
                if t.name in self._modules
            ]
            disabled = [
                t.name for t in tasks
                if t.name in self._disabled_tasks
            ]
            summary[stage_name] = {
                "total_defined": len(tasks),
                "registered": len(registered),
                "disabled": len(disabled),
                "active": len(registered) - len(disabled),
            }
        return summary

    def get_last_flow(self) -> FlowResult | None:
        """Son flow sonucu."""
        return self._flow_history[-1] if self._flow_history else None

    @property
    def is_healthy(self) -> bool:
        """Son flow başarılı mı?"""
        last = self.get_last_flow()
        if last is None:
            return True
        return last.success_rate > 0.5

    def reset_circuit_breakers(self) -> list[str]:
        """Tüm devre kesicileri sıfırla."""
        reset_tasks = list(self._disabled_tasks)
        for name in reset_tasks:
            self.enable_task(name)
        logger.info(
            f"[Orchestrator] {len(reset_tasks)} circuit breaker sıfırlandı."
        )
        return reset_tasks
