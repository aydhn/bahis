================================================================================
  QUANT BETTING BOT - EKOSISTEM OZELLIK LISTESI & KULLANIM KILAVUZU
  Versiyon : 1.0
  Tarih    : 2026-02-16
================================================================================

  !! GUNCELLEME ZORUNLULUGU !!
  Kodda yapilan her degisiklikte (yeni modul, parametre degisikligi, yeni
  komut, silinen ozellik, API degisikligi) bu belge de guncellenmelidir.
  Guncellenmesi gereken bolumu bulun, tarihi ekleyin ve degisikligi yazin.
  Ornek: "[2026-03-01] Yeni modul eklendi: src/quant/yeni_modul.py"

================================================================================
ICINDEKILER
================================================================================

  1.  GENEL BAKIS VE MIMARI
  2.  KURULUM VE BASLANGIC
  3.  CLI KOMUTLARI (bahis.py)
  4.  KATMAN 0 - ALTYAPI (Infrastructure)
  5.  KATMAN 1 - VERI TOPLAMA (Sensors & Ingestion)
  6.  KATMAN 2 - HAFIZA VE BAGLAM (Memory & Context)
  7.  KATMAN 3 - KANTITATIF ZEKA (Quantitative Brain)
  8.  KATMAN 4 - RISK VE ICRA (Risk & Execution)
  9.  KATMAN 5 - YARDIMCI ARACLAR VE ARAYUZ (Utils & UI)
  10. ANA ANALIZ DONGUSU (_analysis_loop)
  11. ZAMANLANMIS GOREVLER (Scheduler)
  12. YAPILANDIRMA DOSYALARI
  13. LOG YONETIMI
  14. HATA AYIKLAMA VE IZLEME
  15. BAGIMLILIKLAR VE TEKNOLOJILER
  16. GUNCELLEME KAYITLARI

================================================================================
1. GENEL BAKIS VE MIMARI
================================================================================

Quant Betting Bot, otonom coklu ajan hedge fund mimarisine sahip bir
bahis analiz ve karar destek sistemidir.

Dosya yapisi:
  bahis.py              Ana orkestrator (is mantigi SIFIR, sadece koordinasyon)
  src/
    core/               Cekirdek altyapi ve risk modulleri (33 dosya)
    quant/              Kantitatif analiz ve ML modelleri (60+ dosya)
    ingestion/          Veri toplama ve scraping (12 dosya)
    memory/             Veritabani, cache ve vektör deposu (8 dosya)
    utils/              Yardimci araclar, raporlama, Telegram (20 dosya)
    ui/                 Kullanici arayuzleri (6 dosya)
  logs/                 Guncel log dosyalari (son 3 gun)
  logs/archive/         Arsivlenmis eski loglar (kalici, silinmez)
  data/                 Veri dosyalari
  models/               Egitilmis model dosyalari
  .env                  Gizli ortam degiskenleri (token, sifre vb.)

Toplam: ~150 Python modulu, 444 kurulu kutuphane

Mimari katmanlar:
  KATMAN 0  Infrastructure      Devre kesici, cache, log, hata yonetimi
  KATMAN 1  Sensors/Ingestion   Web scraping, API, vizyon, ses
  KATMAN 2  Memory/Context      DB, vektör arama, graf veritabani
  KATMAN 3  Quantitative Brain  30+ istatistiksel/ML model
  KATMAN 4  Risk/Execution      Portfolio optimizasyonu, Kelly, hedge
  KATMAN 5  Utils/UI            Telegram, raporlama, dashboard

================================================================================
2. KURULUM VE BASLANGIC
================================================================================

Gereksinimler:
  - Python 3.11+
  - Visual Studio Build Tools 2022 (C++ derleyici)
  - CMake 4.x
  - Rust 1.93+ (Cargo dahil)
  - Git

Kurulum adimlari:

  1) Sanal ortam olustur:
     python -m venv .venv

  2) Aktive et:
     Windows:  .venv\Scripts\activate
     Linux:    source .venv/bin/activate

  3) Bagimliliklari kur:
     pip install -r requirements.txt

  4) .env dosyasini yapilandir:
     TELEGRAM_BOT_TOKEN=<bot_token>
     TELEGRAM_CHAT_ID=<chat_id>
     NEO4J_URI=bolt://localhost:7687  (opsiyonel)
     NEO4J_USER=neo4j                (opsiyonel)
     NEO4J_PASSWORD=<sifre>          (opsiyonel)

  5) Botu baslat:
     python bahis.py run

================================================================================
3. CLI KOMUTLARI (bahis.py)
================================================================================
!! Bu bolum yeni @app.command() eklendiginde guncellenmelidir !!

--- ANA KOMUTLAR ---

  python bahis.py run [--mode full|live|pre] [--headless] [--telegram] [--dashboard]
    Botu baslatir ve tum katmanlari ayaga kaldirir.
    --mode full    : Hem pre-match hem live analiz (varsayilan)
    --mode live    : Sadece canli mac analizi
    --mode pre     : Sadece on-mac analizi
    --headless     : Tarayiciyi gorunmez modda calistir (varsayilan: True)
    --telegram     : Telegram botunu baslat
    --dashboard    : Terminal dashboard (TUI) ac

  python bahis.py backtest [--start 2024-01-01] [--end 2026-01-01]
    Gecmis veri uzerinde strateji testi calistirir.

  python bahis.py report
    PDF formatinda strateji saglik raporu uretir.

  python bahis.py doctor
    Sistem bilesen saglik kontrolu yapar, eksik bagimliliklari tespit eder.

  python bahis.py docs
    Otomatik API dokumantasyonunu gunceller.

  python bahis.py web
    Streamlit web dashboard'u baslatir (tarayici acilir).

--- ANALIZ KOMUTLARI ---

  python bahis.py analyze <ev_takimi> <deplasman_takimi> [--home-xg 1.4] [--away-xg 1.1]
    Tek bir maci derinlemesine analiz eder.
    Poisson + Dixon-Coles + Monte Carlo + Elo + Gradient Boosting + VIX

  python bahis.py nash <ev_takimi> <deplasman_takimi>
    Nash Equilibrium ile optimal strateji hesaplar.

  python bahis.py entropy <ev_takimi> <deplasman_takimi>
    Shannon Entropy ve KL-Divergence analizi.

  python bahis.py chaos <ev_takimi> <deplasman_takimi>
    Lyapunov Exponent ile kaos analizi.

  python bahis.py hawkes <ev_takimi> <deplasman_takimi>
    Hawkes Process ile momentum ve bulasicilik analizi.

  python bahis.py survival <ev_takimi> <deplasman_takimi>
    Survival Analysis ile gol beklenti zamani.

  python bahis.py fatigue <ev_takimi> <deplasman_takimi>
    Biyomekanik yorgunluk modeli.

  python bahis.py wavelet <ev_takimi> <deplasman_takimi>
    Wavelet Transform ile sinyal analizi.

  python bahis.py homology <ev_takimi> <deplasman_takimi>
    Homology ile takim organizasyon analizi.

  python bahis.py topology <ev_takimi> <deplasman_takimi>
    TDA (Topological Data Analysis) analizi.

  python bahis.py volatility <ev_takimi> <deplasman_takimi>
    GARCH ile volatilite analizi.

  python bahis.py particle <ev_takimi> <deplasman_takimi>
    Particle Filter ile dinamik guc takibi.

  python bahis.py causal <ev_takimi> <deplasman_takimi>
    Nedensellik kesfii ve dogrudan etki analizi.

  python bahis.py probabilistic <ev_takimi> <deplasman_takimi>
    PyMC ile Bayesian olasiliksal tahmin.

  python bahis.py multifractal <ev_takimi> <deplasman_takimi>
    MF-DFA ile coklu fraktal analiz.

  python bahis.py symbolic <ev_takimi> <deplasman_takimi>
    PySR ile sembolik regresyon (formul kesfii).

  python bahis.py fuzzy <ev_takimi> <deplasman_takimi>
    Fuzzy Logic ile bulanik mantik degerlendirmesi.

  python bahis.py uncertainty <ev_takimi> <deplasman_takimi>
    Epistemic vs Aleatoric belirsizlik ayrimi.

  python bahis.py decision-flow <ev_takimi> <deplasman_takimi>
    Karar akisi gorsellestirmesi (flowchart).

--- YONETIM KOMUTLARI ---

  python bahis.py regime-kelly
    Regime-Aware Kelly Criterion durumunu gosterir.

  python bahis.py fisher
    Fisher Information Geometry raporu.

  python bahis.py philo
    Epistemik Muhakeme Motoru raporu.

  python bahis.py evolver
    Strateji Evrim (Genetik Algoritma) durumu.

  python bahis.py guardian
    Exception Guardian saglik raporu.

  python bahis.py super-log
    SuperLogger istatistikleri.

  python bahis.py stream
    Stream Processor canli durum bilgisi.

  python bahis.py orchestrator
    Workflow Orchestrator pipeline durumu.

  python bahis.py agent-poll <ev_takimi> <deplasman_takimi>
    Coklu ajan oylama sistemi.

  python bahis.py graphrag <ev_takimi> <deplasman_takimi>
    GraphRAG kriz analizi.

  python bahis.py active-inf <ev_takimi> <deplasman_takimi>
    Active Inference ajan raporu.

  python bahis.py automl
    AutoML model arama (TPOT/RandomizedSearch).

  python bahis.py synthetic
    Sentetik veri uretimi.

================================================================================
4. KATMAN 0 - ALTYAPI (Infrastructure)
================================================================================
!! Bu bolum src/core/ altinda degisiklik yapildiginda guncellenmelidir !!

4.1  CircuitBreakerRegistry          src/core/circuit_breaker.py
     ---------------------------------------------------------------
     Amac   : Hata toleransi ve devre kesici deseni.
     Sinif  : CircuitBreaker, CircuitBreakerRegistry, ModuleLoader
     Kullanim:
       registry = CircuitBreakerRegistry()
       cb = registry.get_or_create("scraper", preset="scraper")
       result = cb.call(fonksiyon, arguman1, arguman2)
       # Asenkron: result = await cb.call_async(fonksiyon, arg1)
     Presetler: "scraper", "model", "api", "db"
     Davranis: threshold asildiginda devre acar, bekleme suresi sonra
               yarim-acik duruma gecer ve tek istek ile test eder.
     Durum:    registry.all_statuses()  # tum devrelerin durumu
               registry.open_breakers() # acik devreler
               registry.reset_all()     # tum devreleri sifirla

4.2  SmartCache                      src/memory/smart_cache.py
     ---------------------------------------------------------------
     Amac   : Cok katmanli onbellek (L1 LRU, L2 TTL RAM, L3 Disk).
     Sinif  : TTLCache, SmartCache
     Kullanim:
       cache = SmartCache(ttl_l2=3600.0, ttl_l3=86400.0)
       sonuc = cache.get_or_compute("anahtar", hesaplama_fonksiyonu)
     Dekorator:
       @cached(ttl=300)
       def agir_hesaplama(x): ...
     Istatistik: cache.stats  # hit_rate, miss_count vb.

4.3  SuperLogger                     src/utils/super_logger.py
     ---------------------------------------------------------------
     Amac   : Yapilandirilmis JSON loglama, modul bazli izleme.
     Sinif  : SuperLogger (Singleton)
     Kullanim:
       slog = SuperLogger()
       ml = slog.get_module_logger("poisson")
       ml.info("Tahmin tamamlandi")
       with slog.timed("poisson"):
           # zamanlanan islem
       slog.log_decision(modul="kelly", karar="BET", detay={...})
     Metodlar:
       get_slowest_modules(top_n=5)     # en yavas moduller
       get_error_prone_modules(top_n=5) # en cok hata veren moduller
       get_module_stats("poisson")      # modul istatistikleri

4.4  LogRotator                      src/utils/log_rotator.py
     ---------------------------------------------------------------
     Amac   : Eski log dosyalarini arsivleme (silinmez, kalici).
     Sinif  : LogRotator
     Kullanim:
       rotator = LogRotator(log_dir="logs", archive_days=3, compress=True)
       rapor = rotator.rotate()  # eski loglari arsivle
     Otomatik: Her gece 02:00'da scheduler ile calisir.
     Arsiv  : logs/archive/ altinda gz olarak saklanir.
     Not    : Arsivlenen loglar ASLA silinmez.

4.5  ExceptionGuardian               src/core/exception_guardian.py
     ---------------------------------------------------------------
     Amac   : Sessiz hata yakalama, siniflandirma, devre kesici.
     Sinif  : ExceptionGuardian, ExceptionTaxonomy
     Kullanim:
       guardian = ExceptionGuardian(error_budget_per_hour=50)
       with guardian.protect("modul_adi"):
           # korunacak islem
       # veya dekorator:
       @guardian.guard("modul_adi")
       def islem(): ...
     Siniflandirma: network, data, model, system, unknown
     Rapor  : guardian.health_report()
     Hata   : guardian.get_recent_errors(limit=20)
     Kalp   : guardian.heartbeat("modul_adi")
              guardian.check_heartbeats(timeout=300)

4.6  RegimeKelly                     src/core/regime_kelly.py
     ---------------------------------------------------------------
     Amac   : Rejim-farkinda Kelly Criterion ile stake hesaplama.
     Sinif  : RegimeKelly
     Kullanim:
       rk = RegimeKelly(bankroll=10000, base_fraction=0.25,
                         min_edge=0.03, max_stake_pct=0.05)
       karar = rk.calculate(probability=0.55, odds=2.10,
                             regime=RegimeState(...))
       # karar.stake, karar.edge, karar.kelly_fraction
       rk.record_result(won=True, pnl=50.0)
     Ozellikler:
       - Bankroll segmentasyonu
       - Anti-tilt mekanizmasi (5 kayip sonrasi otomatik durdurma)
       - Gunluk exposure limiti (%15)
       - Volatilite hedefleme
       - Drawdown protokolu

4.7  StrategyEvolver                 src/core/strategy_evolver.py
     ---------------------------------------------------------------
     Amac   : Genetik algoritma ile strateji otomatik evrimi.
     Sinif  : StrategyEvolver, StrategyDNA
     Kullanim:
       evolver = StrategyEvolver(population_size=50, epoch_size=100)
       rapor = evolver.evolve(sonuclar)  # son sonuclarla evrim
       en_iyi = evolver.get_best_dna()   # en iyi strateji DNA'si
     Ozellikler:
       - Popülasyon: 50 birey
       - Elitizm: %10
       - Turnuva secimi, caprazlama, mutasyon
       - Hall of Fame (en iyi 5)
       - Checkpoint kaydetme/yukleme

4.8  JobScheduler                    src/core/job_scheduler.py
     ---------------------------------------------------------------
     Amac   : APScheduler tabanli gorev zamanlayici.
     Sinif  : JobScheduler
     Kullanim:
       scheduler = JobScheduler(timezone="Europe/Istanbul")
       scheduler.add_cron("log_rotate", rotator.rotate, hour=2, minute=0)
       scheduler.add_interval("saglik", kontrol, minutes=5)
       await scheduler.start()
     Metodlar: add_cron(), add_interval(), add_date(), get_jobs()

4.9  EventBus                       src/core/event_bus.py
     ---------------------------------------------------------------
     Amac   : Event-driven mimari, event saklama ve tekrarlama.
     Sinif  : EventStore, EventBus, ReplayEngine
     Kullanim:
       store = EventStore()
       bus = EventBus(store=store)
       bus.subscribe("goal", goal_handler)
       await bus.emit(Event(event_type="goal", data={...}))
       # Tekrarlama:
       replay = ReplayEngine(bus)
       await replay.replay_match("match_123", speed=10.0)

4.10 WorkflowOrchestrator           src/core/workflow_orchestrator.py
     ---------------------------------------------------------------
     Amac   : Prefect tabanli pipeline orkestrasyon.
     Sinif  : WorkflowOrchestrator
     Kullanim:
       orch = WorkflowOrchestrator(max_retries=3, task_timeout=120)
       orch.register_modules({"poisson": poisson, "mc": mc})
       sonuc = await orch.run_pipeline(context={...})
     Asamalar: ingestion -> quant -> risk -> execution -> reporting

4.11 StreamProcessor                 src/core/stream_processor.py
     ---------------------------------------------------------------
     Amac   : Gercek zamanli olay akisi isleme.
     Sinif  : StreamProcessor, WindowedBuffer
     Kullanim:
       sp = StreamProcessor(window_sec=60.0, n_workers=4)
       sp.register_consumer("odds", odds_handler)
       await sp.start()
       await sp.emit(StreamEvent(match_id="m1", value=2.1))

4.12 DistributedCore                 src/core/distributed_core.py
     ---------------------------------------------------------------
     Amac   : Ray tabanli dagitik hesaplama.
     Sinif  : DistributedCore
     Kullanim:
       dc = DistributedCore(num_cpus=4, max_workers=8)
       dc.start()
       ref = dc.submit_monte_carlo(home_xg=1.5, away_xg=0.8)
       sonuclar = dc.gather([ref])

4.13 GRPCCommunicator               src/core/grpc_communicator.py
     ---------------------------------------------------------------
     Amac   : gRPC ve in-process bus ile mikroservis iletisimi.
     Sinif  : GRPCCommunicator, InProcessBus
     Kullanim:
       comm = GRPCCommunicator(use_grpc=False)  # in-process mod
       await comm.start()
       await comm.send("odds", ServiceMessage(data={...}))
       msg = await comm.receive("odds", timeout=5.0)

4.14 TelemetryTracer                 src/core/telemetry_tracer.py
     ---------------------------------------------------------------
     Amac   : OpenTelemetry tabanli dagitik izleme.
     Sinif  : TelemetryTracer, SimpleProfiler
     Kullanim:
       tracer = TelemetryTracer(service_name="quant-betting-bot")
       with tracer.span("poisson_predict", module="poisson"):
           # izlenen islem
       rapor = tracer.get_bottleneck_report(top_n=10)

4.15 DependencyContainer             src/core/dependency_container.py
     ---------------------------------------------------------------
     Amac   : Dependency injection container.
     Sinif  : DependencyContainer
     Kullanim:
       container = DependencyContainer()
       container.register("db", lambda: DBManager(), singleton=True)
       db = container.resolve("db")

4.16 DataValidator                   src/core/data_validator.py
     ---------------------------------------------------------------
     Amac   : Pydantic tabanli veri dogrulama.
     Sinif  : DataValidator
     Kullanim:
       validator = DataValidator()
       temiz = validator.validate_batch(kirli_veri, schema="match")

4.17 RustEngine                      src/core/rust_engine.py
     ---------------------------------------------------------------
     Amac   : Rust/Numba ile yuksek performans hesaplamalar.
     Sinif  : RustEngine
     Kullanim:
       engine = RustEngine()
       sonuc = engine.monte_carlo_sim(n_sims=100000)
       kelly = engine.kelly_batch(probs, odds)
     Fallback: Rust modulu yoksa Numba JIT kullanir.

4.18 JITAccelerator                  src/core/jit_accelerator.py
     ---------------------------------------------------------------
     Amac   : Numba JIT ile kritik hesaplamalari hizlandirma.
     Sinif  : JITAccelerator, ArrowBridge
     Kullanim:
       jit = JITAccelerator()
       jit.warmup()  # on isitma
       sonuc = jit.kelly(prob=0.55, odds=2.1)
       matris = jit.poisson_matrix(1.5, 0.8)

4.19 JAXAccelerator                  src/core/jax_accelerator.py
     ---------------------------------------------------------------
     Amac   : JAX/XLA ile GPU hizlandirma.
     Sinif  : JAXAccelerator
     Kullanim:
       jax_acc = JAXAccelerator()
       hizli_df = jax_acc.accelerate(features_df)

4.20 BlindStrategyEngine             src/core/blind_strategy.py
     ---------------------------------------------------------------
     Amac   : Homomorfik sifreleme ile gizli strateji hesaplama.
     Sinif  : BlindStrategyEngine, MaskedCompute
     Kullanim:
       bse = BlindStrategyEngine()
       enc = bse.encrypt([0.55, 0.25, 0.20])
       sonuc = bse.blind_kelly(enc, probs, odds)
     Bagimlilik: TenSEAL (opsiyonel, fallback: maskeleme)

4.21 ShadowManager                   src/core/shadow_manager.py
     ---------------------------------------------------------------
     Amac   : Shadow trading / paper trading sistemi.
     Sinif  : ShadowManager
     Kullanim:
       sm = ShadowManager()
       sm.register_strategy("agresif", bankroll=10000, live=False)
       sm.place_bet("agresif", "m1", "home", 2.1, 100)
       sm.settle("m1", "home", won=True)
       rapor = sm.compare_all()  # tum stratejileri karsilastir

4.22 MimicEngine                     src/core/mimic_engine.py
     ---------------------------------------------------------------
     Amac   : Insansi davranis simulasyonu (anti-ban).
     Sinif  : MimicEngine
     Kullanim:
       mimic = MimicEngine(persona="cautious")
       yol = mimic.mouse_path(baslangic, bitis)
       await mimic.human_delay(action="click")
     Personalar: "fast_bettor", "cautious", "researcher", "random"

4.23 SelfHealingEngine               src/core/auto_healer.py
     ---------------------------------------------------------------
     Amac   : LLM ile otomatik kod onarimi.
     Sinif  : SelfHealingEngine, HealingLog
     Kullanim:
       healer = SelfHealingEngine(llm_backend="ollama")
       sonuc = await healer.attempt_heal(exception, module_path="...")
       healer.rollback_last("modul_adi")

4.24 QuantumAnnealer                 src/core/quantum_annealer.py
     ---------------------------------------------------------------
     Amac   : Simulated annealing ile portfoy secimi.
     Sinif  : QuantumAnnealer
     Kullanim:
       qa = QuantumAnnealer(bankroll=10000, max_bets=10, max_risk=0.15)
       cozum = qa.optimize(adaylar)

4.25 ActiveInferenceAgent            src/core/active_inference_agent.py
     ---------------------------------------------------------------
     Amac   : Free Energy Principle ile otonom ogrenme.
     Sinif  : ActiveInferenceAgent
     Kullanim:
       aia = ActiveInferenceAgent(modules=["poisson","mc","elo"])
       surprisal = aia.observe("poisson", [0.5,0.3,0.2], observed=0)
       hedefler = aia.get_retrain_targets()
       agirliklar = aia.get_precision_weights()

4.26 FairValueEngine                 src/core/fair_value_engine.py
     ---------------------------------------------------------------
     Amac   : Gercek deger ve value edge hesaplama.
     Sinif  : FairValueEngine
     Kullanim:
       fve = FairValueEngine(min_edge=0.02, kelly_fraction=0.25)
       sonuc = fve.analyze(model_prob=0.55, market_odds=2.10)
       # sonuc.has_value, sonuc.edge, sonuc.kelly_stake
       vig = fve.remove_vig(1.9, 3.4, 4.2)  # vig kaldirma

4.27 HedgeCalculator                 src/core/hedge_calculator.py
     ---------------------------------------------------------------
     Amac   : Arbitraj ve hedge firsati hesaplama.
     Sinif  : HedgeCalculator
     Kullanim:
       hc = HedgeCalculator(min_profit_pct=0.01)
       surebet = hc.check_surebet(1.9, 3.4, 4.2)
       hedge = hc.calculate_hedge(100, 2.1, "home", canli_oranlar)
       firsatlar = hc.scan_active_bets(aktif_bahisler, canli_oranlar)

4.28 PortfolioOptimizer              src/core/portfolio_optimizer.py
     ---------------------------------------------------------------
     Amac   : Markowitz portfoy optimizasyonu.
     Sinif  : PortfolioOptimizer
     Kullanim:
       po = PortfolioOptimizer(initial_bankroll=10000, max_portfolio_risk=0.15)
       tahsis = po.optimize(adaylar)  # risk-getiri dengesi

4.29 BlackLittermanOptimizer         src/core/black_litterman_optimizer.py
     ---------------------------------------------------------------
     Amac   : Black-Litterman portfoy optimizasyonu.
     Sinif  : BlackLittermanOptimizer
     Kullanim:
       blo = BlackLittermanOptimizer(risk_aversion=2.5, tau=0.05)
       sonuc = blo.optimize(ensemble, risk_metrics)

4.30 ConstrainedRiskSolver           src/core/constrained_risk_solver.py
     ---------------------------------------------------------------
     Amac   : Lagrange carpanlari ile kisitli risk optimizasyonu.
     Sinif  : ConstrainedRiskSolver
     Kullanim:
       crs = ConstrainedRiskSolver(max_single_stake=0.05,
                                    max_total_exposure=0.20)
       tahsis = crs.solve(aday_listesi)
       dogrulama = crs.validate(bahis_listesi)

4.31 SystemicRiskCoVaR               src/core/systemic_risk_covar.py
     ---------------------------------------------------------------
     Amac   : CoVaR ile sistemik risk olcumu.
     Sinif  : SystemicRiskCoVaR
     Kullanim:
       covar = SystemicRiskCoVaR(confidence_level=0.95)
       risk = covar.measure(ensemble)
       corr = covar.correlation_matrix(ensemble)

4.32 PnLStabilizer                   src/core/pnl_stabilizer.py
     ---------------------------------------------------------------
     Amac   : PID kontrolcusu ile PnL egrisi stabilizasyonu.
     Sinif  : PnLStabilizer
     Kullanim:
       pnl = PnLStabilizer(target_daily_return=0.002)
       duzeltilmis = pnl.stabilize(bahisler)
       pnl.record_pnl(bankroll=10500)

4.33 GeneticOptimizer                src/core/genetic_optimizer.py
     ---------------------------------------------------------------
     Amac   : Parametre optimizasyonu icin genetik algoritma.
     Sinif  : GeneticOptimizer
     Kullanim:
       go = GeneticOptimizer(population_size=100, mutation_rate=0.15)
       en_iyi = go.evolve(backtest_fn, generations=50)
       go.save_config(en_iyi)

4.34 VectorBacktester                src/core/vector_backtester.py
     ---------------------------------------------------------------
     Amac   : Vektorizel backtesting motoru.
     Sinif  : VectorBacktester
     Kullanim:
       bt = VectorBacktester(initial_bankroll=10000)
       sonuclar = bt.run(db, start="2024-01-01", end="2026-01-01")

4.35 ModelQuantizer                  src/core/model_quantizer.py
     ---------------------------------------------------------------
     Amac   : Model sikistirma (INT8/FP16 quantization).
     Sinif  : ModelQuantizer
     Kullanim:
       mq = ModelQuantizer()
       kucuk = mq.quantize_torch_model(model, method="dynamic")

================================================================================
5. KATMAN 1 - VERI TOPLAMA (Sensors & Ingestion)
================================================================================
!! Bu bolum src/ingestion/ altinda degisiklik yapildiginda guncellenmelidir !!

5.1  DataFactory                     src/ingestion/async_data_factory.py
     ---------------------------------------------------------------
     Amac   : Asenkron veri toplama fabrikasi.
     Sinif  : DataFactory
     Kullanim:
       df = DataFactory(db=db, cache=cache, headless=True)
       await df.run_prematch(shutdown)  # on-mac verisi
       await df.run_live(shutdown)      # canli veri
     Kaynaklar: OddsAPI, FootballData, FlashScore
     Ozellikler: Playwright scraping, httpx API, otomatik normalize

5.2  APIHijacker                     src/ingestion/api_hijacker.py
     ---------------------------------------------------------------
     Amac   : XHR/WebSocket trafigini yakalama.
     Sinif  : APIHijacker
     Kullanim:
       hijacker = APIHijacker(db=db)
       await hijacker.listen(shutdown)
       # Yakalanan endpointler: hijacker.discovered_endpoints
       # Dogrudan erisim: await hijacker.direct_fetch(endpoint)

5.3  ScraperAgent                    src/ingestion/scraper_agent.py
     ---------------------------------------------------------------
     Amac   : Mackolik, Sofascore, Transfermarkt scraping.
     Sinif  : ScraperAgent, MackolikScraper, SofascoreScraper,
              TransfermarktScraper
     Kullanim:
       agent = ScraperAgent(db=db, notifier=notifier)
       await agent.run_all(shutdown)
     Ozellikler: Circuit breaker korumasi, rastgele gecikme,
                 User-Agent rotasyonu

5.4  DataSourceAggregator            src/ingestion/data_sources.py
     ---------------------------------------------------------------
     Amac   : Ucretsiz veri kaynaklarini birlestirme.
     Sinif  : DataSourceAggregator, UnderstatSource, FBrefSource,
              FootballCSVSource, SofascoreHiddenAPI
     Kullanim:
       agg = DataSourceAggregator(db=db)
       await agg.fetch_all(league="super_lig", season="2526")
       await agg.fetch_today()

5.5  LineupMonitor                   src/ingestion/lineup_monitor.py
     ---------------------------------------------------------------
     Amac   : Kadro degisikliklerini izleme.
     Sinif  : LineupMonitor
     Kullanim:
       lm = LineupMonitor(db=db, notifier=notifier)
       lm.register_star_players("Galatasaray", ["Icardi","Mertens"])
       await lm.watch(shutdown)
     Ozellikler: 2 dakikada bir kontrol, yildiz oyuncu eksikligi tespiti

5.6  NewsRAGAnalyzer                 src/ingestion/news_rag.py
     ---------------------------------------------------------------
     Amac   : Haber tabanli sentiment analizi (RAG).
     Sinif  : NewsRAGAnalyzer
     Kullanim:
       rag = NewsRAGAnalyzer()
       sonuc = await rag.analyze_match("Galatasaray", "Fenerbahce")
       # sonuc.home.sentiment_score, sonuc.away.sentiment_score
     LLM: Gemini veya HuggingFace (otomatik fallback)

5.7  VisionTracker                   src/ingestion/vision_tracker.py
     ---------------------------------------------------------------
     Amac   : YOLO + OpenCV ile canli mac video analizi.
     Sinif  : VisionTracker
     Kullanim:
       vt = VisionTracker()
       analiz = vt.process_frame(frame)
       heatmap = vt.get_heatmap()

5.8  LiveMatchVision                 src/ingestion/vision_live.py
     ---------------------------------------------------------------
     Amac   : Gercek zamanli YOLO ile canli mac analizi.
     Sinif  : LiveMatchVision, YOLOFootballDetector
     Kullanim:
       lmv = LiveMatchVision(model_size="n", fps=1.0)
       rapor = lmv.get_momentum_report("match_123")
       await lmv.watch_stream("rtmp://...", "match_123", shutdown)
     Sinyaller: GOAL_SMELL, HIGH_PRESSURE, COUNTER, NEUTRAL

5.9  VoiceInterrogator               src/ingestion/voice_interrogator.py
     ---------------------------------------------------------------
     Amac   : Whisper ile sesli komut isleme.
     Sinif  : VoiceInterrogator
     Kullanim:
       vi = VoiceInterrogator(model_size="base")
       metin = vi.transcribe(ses_dosyasi)
       komut = vi.parse_command(metin)

5.10 MetricExporter                  src/ingestion/metric_exporter.py
     ---------------------------------------------------------------
     Amac   : Prometheus metrikleri.
     Sinif  : MetricExporter
     Kullanim:
       me = MetricExporter(port=9090)
       me.serve()  # HTTP endpoint baslatir
       @me.track_latency
       def model_predict(): ...

5.11 StealthBrowser                  src/ingestion/stealth_browser.py
     ---------------------------------------------------------------
     Amac   : Anti-detection tarayici.
     Sinif  : StealthBrowser
     Kullanim:
       sb = StealthBrowser(headless=True)
       await sb.start()
       await sb.goto("https://...", wait_ms=3000)
     Motorlar: undetected-chromedriver, Playwright stealth

5.12 AutoHealer                      src/ingestion/auto_healer.py
     ---------------------------------------------------------------
     Amac   : Eksik bagimlilik tespiti ve otomatik kurulum.
     Sinif  : AutoHealer
     Kullanim:
       healer = AutoHealer()
       healer.diagnose()  # eksikleri goster
       healer.heal()      # otomatik duzelt

================================================================================
6. KATMAN 2 - HAFIZA VE BAGLAM (Memory & Context)
================================================================================
!! Bu bolum src/memory/ altinda degisiklik yapildiginda guncellenmelidir !!

6.1  DBManager                       src/memory/db_manager.py
     ---------------------------------------------------------------
     Amac   : Merkezi veritabani (Polars + DuckDB).
     Sinif  : DBManager
     Kullanim:
       db = DBManager()
       maclar = db.get_upcoming_matches(hours_ahead=48)
       takim = db.get_team_stats("Galatasaray")
       oranlar = db.get_odds_history("match_123")
       matris = db.build_feature_matrix(maclar)
       db.upsert_match({...})
       db.save_signals(sinyaller, cycle=1)
       sonuc = db.query("SELECT * FROM matches WHERE ...")
     Tablolar: matches, signals, historical_stats, odds_history

6.2  FeatureCache                    src/memory/feature_cache.py
     ---------------------------------------------------------------
     Amac   : DiskCache tabanli SSD onbellekleme.
     Sinif  : FeatureCache
     Kullanim:
       cache = FeatureCache(size_limit_gb=2.0)
       sonuc = cache.get_or_compute("key", hesapla_fn, ttl=300)
       @cache.memoize(ttl=300)
       def agir_hesaplama(x): ...

6.3  LanceMemory                     src/memory/lance_memory.py
     ---------------------------------------------------------------
     Amac   : LanceDB vektor veritabani (semantik arama).
     Sinif  : LanceMemory
     Kullanim:
       lance = LanceMemory()
       lance.add("doc1", "Galatasaray baskan degisikligi", category="news")
       lance.add_odds_event("m1", "1x2", {"home":1.9}, "mackolik")
       lance.add_news("baslik", "icerik", "source")
       benzerler = lance.search("takim krizi", top_k=5)

6.4  GraphRAG                        src/memory/graph_rag.py
     ---------------------------------------------------------------
     Amac   : Neo4j + LLM ile bilgi grafi ve kriz analizi.
     Sinif  : GraphRAG
     Kullanim:
       grag = GraphRAG(llm_backend="auto")
       grag.ingest_news(haberler, team="Galatasaray")
       rapor = grag.analyze_crisis("Galatasaray", lookback_hours=48)
       cevap = grag.ask("Galatasaray'in son haftadaki performansi?")
     Fallback: Neo4j yoksa NetworkX kullanir.

6.5  Neo4jFootballGraph              src/memory/neo4j_graph.py
     ---------------------------------------------------------------
     Amac   : Neo4j graf veritabani (takimlar, oyuncular, hakemler).
     Sinif  : Neo4jFootballGraph
     Kullanim:
       graph = Neo4jFootballGraph(uri="bolt://localhost:7687")
       graph.create_match(mac_verisi)
       bias = graph.query_referee_bias("hakem_adi")
       h2h = graph.query_h2h_graph("Galatasaray", "Fenerbahce")
     Fallback: Neo4j yoksa NetworkX kullanir.

6.6  SmartCache                      src/memory/smart_cache.py
     (Detaylar 4.2 numarali maddede)

6.7  ZeroCopyBridge                  src/memory/zero_copy_bridge.py
     ---------------------------------------------------------------
     Amac   : Shared memory ile zero-copy veri paylasimi.
     Sinif  : ZeroCopyBridge
     Kullanim:
       bridge = ZeroCopyBridge()
       meta = bridge.publish("matris", numpy_array)
       array = bridge.subscribe(meta)

6.8  DVCManager                      src/memory/dvc_manager.py
     ---------------------------------------------------------------
     Amac   : DVC ile veri versiyonlama.
     Sinif  : DVCManager
     Kullanim:
       dvc = DVCManager(data_dir="data")
       dvc.init()
       dvc.track("data/matches.parquet")
       dvc.snapshot(tag="v1.0")

================================================================================
7. KATMAN 3 - KANTITATIF ZEKA (Quantitative Brain)
================================================================================
!! Bu bolum src/quant/ altinda degisiklik yapildiginda guncellenmelidir !!

--- 7.1 TEMEL ISTATISTIKSEL MODELLER ---

  PoissonModel             src/quant/poisson_model.py
    Bivariate Poisson model. predict_for_dataframe() ile 1X2, O/U, BTTS.

  DixonColesModel          src/quant/dixon_coles_model.py
    Dixon-Coles duzeltilmis Poisson. Dusuk skorlu beraberlikleri duzeltir.

  MonteCarloEngine         src/quant/monte_carlo_engine.py
    Monte Carlo simulasyonu. xG bazli 1000+ simulasyon.

  EloGlickoSystem          src/quant/elo_glicko_rating.py
    Elo ve Glicko-2 rating sistemi. Dinamik guc tahmini.

  GLMGoalPredictor         src/quant/glm_model.py
    Generalized Linear Models. Poisson veya Negative Binomial.

--- 7.2 MAKINE OGRENMESI MODELLERI ---

  GradientBoostingModel    src/quant/gradient_boosting.py
    LightGBM/XGBoost. 3 sinif tahmini (Home/Draw/Away).

  LSTMTrendAnalyzer        src/quant/lstm_trend.py
    LSTM zaman serisi. Son 10 mac ile momentum analizi.

  MultiTaskBackbone        src/quant/multi_task_backbone.py
    Cok gorevli ogrenme (MTL). PyTorch tabanli.

  KANInterpreter           src/quant/kan_interpreter.py
    Kolmogorov-Arnold Networks. Yorumlanabilir ML.

  TransferLearner          src/quant/transfer_learner.py
    Ligler arasi bilgi transferi.

  FederatedTrainer         src/quant/federated_trainer.py
    Federated Learning. Coklu lig egitimi.

  AutoMLEngine             src/quant/automl_engine.py
    TPOT ile otomatik model arama. 5 generation, 50 population.

  SyntheticTrainer         src/quant/synthetic_trainer.py
    Sentetik veri uretimi ve augmentasyon.

--- 7.3 BAYESIAN VE OLASILIKSAL ---

  BayesianHierarchicalModel  src/quant/bayesian_hierarchical.py
    Bayesian hiyerarsik model. PyMC tabanli.

  ProbabilisticEngine      src/quant/probabilistic_engine.py
    PyMC ile olasiliksal programlama. HDI hesaplama.

  EnsembleStacking         src/quant/ensemble_stacking.py
    Meta-model stacking. Walk-forward validation.

  ConformalQuantileBridge   src/quant/conformal_quantile_bridge.py
    Conformal Quantile Regression. Istatistiksel garantili tahmin.

  UncertaintyQuantifier    src/quant/uncertainty_quantifier.py
    Belirsizlik olcumu. Abstain threshold: 0.50.

  UncertaintySeparator     src/quant/uncertainty_separator.py
    Epistemic vs Aleatoric belirsizlik ayrimi.

--- 7.4 NEDENSELLIK VE GRAF ANALIZI ---

  CausalDiscovery          src/quant/causal_discovery.py
    Nedensel graf kesfii. DoWhy kutuphanesi.

  CausalReasoner           src/quant/causal_reasoner.py
    ATE (Average Treatment Effect). Counterfactual analiz.

  GCNPitchGraph            src/quant/gcn_pitch_graph.py
    Graph Convolutional Networks. Oyuncu pozisyon grafi.

  NetworkCentralityAnalyzer  src/quant/network_centrality.py
    PageRank, Betweenness centrality. Eksik oyuncu etkisi.

--- 7.5 FIZIK VE MATEMATIK MODELLERI ---

  PathSignatureEngine      src/quant/path_signature_engine.py
    Rough Path Theory. iisignature ile zaman serisi imzasi.

  JumpDiffusionModel       src/quant/jump_diffusion_model.py
    Merton Jump-Diffusion. Oran sicramasi tespiti.

  GeometricIntelligence    src/quant/geometric_intelligence.py
    Clifford Algebra. Uzaysal analiz.

  QuantumBrain             src/quant/quantum_brain.py
    Variational Quantum Circuits. PennyLane.

  FisherGeometry           src/quant/fisher_geometry.py
    Fisher Information Matrix. Anomali ve rejim degisimi tespiti.
    Kullanim:
      fg = FisherGeometry(anomaly_threshold=2.0, regime_threshold=1.5)
      rapor = fg.compare_distributions(model_probs, market_probs)
      # rapor.fisher_rao_distance, rapor.is_anomaly, rapor.regime_shift

--- 7.6 ZAMAN SERISI VE REJIM ANALIZI ---

  RegimeSwitcher           src/quant/regime_switcher.py
    Hidden Markov Models. 3 rejim tespiti.

  SDEPricer                src/quant/sde_pricer.py
    Stochastic Differential Equations. Oran tahmini.

  HawkesMomentumAnalyzer   src/quant/hawkes_momentum.py
    Hawkes Process. Gol bulasiciligi ve momentum.

  ProphetSeasonalityAnalyzer  src/quant/prophet_seasonality.py
    Facebook Prophet. Mevsimsellik analizi.

  KalmanTeamTracker        src/quant/kalman_tracker.py
    Kalman Filter. Dinamik takim gucu takibi.

  ParticleStrengthTracker  src/quant/particle_strength_tracker.py
    Particle Filter. 1000 parcacik ile guc takibi.

  SurvivalEstimator        src/quant/survival_estimator.py
    Kaplan-Meier. Gol beklenti zamani.

  FatigueEngine            src/quant/fatigue_engine.py
    Biyomekanik yorgunluk modeli. Savunma cokus tespiti.

  VolatilityAnalyzer       src/quant/volatility_analyzer.py
    GARCH(1,1). VaR hesaplama.

--- 7.7 TOPOLOJI VE KAOS ---

  TopologyScanner          src/quant/topology_scanner.py
    TDA (Topological Data Analysis). giotto-tda.

  TopologyMapper           src/quant/topology_mapper.py
    Mapper algoritma. Anomali tespiti.

  ChaosFilter              src/quant/chaos_filter.py
    Lyapunov Exponent. Kaotik maclarda bahisi durdurma.

  HomologyScanner          src/quant/homology_scanner.py
    Homology. Takim organizasyon analizi.

--- 7.8 FRAKTAL VE DALGA ANALIZI ---

  FractalAnalyzer          src/quant/fractal_analyzer.py
    Hurst Exponent. Trending/mean-reverting/random tespiti.

  MultifractalAnalyzer     src/quant/multifractal_logic.py
    MF-DFA. Coklu fraktal rejim degisimi.

  WaveletDenoiser          src/quant/wavelet_denoiser.py
    Daubechies 4 wavelet. Sahte hareket tespiti.

  SymbolicDiscovery        src/quant/symbolic_discovery.py
    PySR. Sembolik regresyon ile formul kesfii.

--- 7.9 RISK VE ANOMALI ---

  EVTTailScanner           src/quant/evt_tail_scanner.py
    Extreme Value Theory. Black Swan taramasi.

  EVTRiskManager           src/quant/evt_risk_manager.py
    EVT tabanli risk yonetimi. Kelly indirimi %50.

  CopulaRiskAnalyzer       src/quant/copula_risk.py
    Copula fonksiyonlari. Kuyruk bagimliligi.

  IsolationAnomalyDetector  src/quant/isolation_anomaly.py
    Isolation Forest. Tuzak/sike taramasi.

  AnomalyDetector          src/quant/anomaly_detector.py
    Z-Score ve Dropping Odds tespiti.

  TransportMetric          src/quant/transport_metric.py
    Wasserstein Distance. Model drift tespiti.

--- 7.10 DIGER ANALIZ MODULLERI ---

  DigitalTwinSimulator     src/quant/digital_twin_sim.py
    Agent-Based Modeling. 200 simulasyon ile mac simulasyonu.

  EntropyMeter             src/quant/entropy_meter.py
    Shannon Entropy. Kill switch threshold: 2.50.

  NashGameSolver           src/quant/nash_solver.py
    Nash Equilibrium. Optimal strateji.

  VectorMatchEngine        src/quant/vector_engine.py
    FAISS vektor veritabani. Tarihsel ikiz bulma.

  SentimentAnalyzer        src/quant/sentiment_analyzer.py
    VADER + TextBlob. Turkce lexicon.

  XAIExplainer             src/quant/xai_explainer.py
    SHAP. Ozellik katkisi analizi.

  PhilosophicalEngine      src/quant/philosophical_engine.py
    Epistemik muhakeme motoru.
    Kullanim:
      pe = PhilosophicalEngine(calibration_window=200, min_score=0.45)
      rapor = pe.evaluate(bahis_listesi)
      # Dunning-Kruger, Black Swan, Lindy, Falsifiability filtreleri

  FuzzyReasoningEngine     src/quant/fuzzy_reasoning.py
    Fuzzy Logic. Bulanik mantik risk degerlendirmesi.

  RLBettingAgent           src/quant/rl_betting_env.py
    Reinforcement Learning. PPO ile bahis karari.

  RLTrader                 src/quant/rl_trader.py
    RL tabanli trader. Stable-baselines3.

  TimeDecay                src/quant/time_decay.py
    Zaman agirliklama. Exponential/logarithmic decay.

  CLVTracker               src/quant/clv_tracker.py
    Closing Line Value takibi. Uzun vadeli karlilik.

  HypergraphUnitAnalyzer   src/quant/hypergraph_unit.py
    Hypergraph Neural Networks. Taktiksel birim analizi.

  FluidPitchAnalyzer       src/quant/fluid_pitch.py
    Fluid Dynamics. Saha kontrol analizi.

  RicciFlowAnalyzer        src/quant/ricci_flow.py
    Ricci Flow. Geometrik donusum.

  BSTSImpactAnalyzer       src/quant/bsts_impact.py
    Bayesian Structural Time Series. Yapisal kirilma tespiti.

================================================================================
8. KATMAN 4 - RISK VE ICRA (Risk & Execution)
================================================================================

Bu katman Katman 0'daki risk modullerini kullanir:
  - RegimeKelly (4.6)        : Rejim-farkinda stake hesaplama
  - FairValueEngine (4.26)   : Deger analizi
  - HedgeCalculator (4.27)   : Arbitraj ve hedge
  - PortfolioOptimizer (4.28): Portfoy optimizasyonu
  - BlackLitterman (4.29)    : BL portfoy optimizasyonu
  - ConstrainedRisk (4.30)   : Kisitli risk optimizasyonu
  - SystemicRisk (4.31)      : Sistemik risk olcumu
  - PnLStabilizer (4.32)     : PnL stabilizasyonu
  - QuantumAnnealer (4.24)   : Simulated annealing
  - ShadowManager (4.21)     : Paper trading

Karar zinciri:
  1. Model tahminleri -> Ensemble birlestirme
  2. FairValue analizi -> Value edge kontrolu
  3. Risk filtreleri (EVT, Copula, Entropy, Chaos)
  4. Portfoy optimizasyonu (BL, Constrained)
  5. Kelly Criterion (Regime-Aware)
  6. PnL stabilizasyonu
  7. Shadow test (paper trade) veya gercek bahis

================================================================================
9. KATMAN 5 - YARDIMCI ARACLAR VE ARAYUZ (Utils & UI)
================================================================================
!! Bu bolum src/utils/ ve src/ui/ altinda degisiklik yapildiginda
   guncellenmelidir !!

--- 9.1 TELEGRAM MODULLERI ---

  TelegramNotifier/TelegramApp   src/ui/telegram_mini_app.py
    Ana Telegram bot. Komutlar, bildirimler, inline butonlar.
    Komutlar:
      /start     - Bot baslatma
      /help      - Yardim
      /durum     - Sistem durumu
      /fikstur   - Yaklasan maclar
      /signals   - Son sinyaller
      /report    - Gunluk rapor
      /clv       - Closing Line Value
      /devreler  - Circuit breaker durumu
      /log       - Son loglar
      /volatility - Volatilite raporu
      /hitl      - Human-in-the-loop kararlari
      /portfoy   - Portfoy durumu

  StrategyCockpit             src/utils/strategy_cockpit.py
    Canli strateji HUD. Tek mesajda tum sistem durumu.

  TelegramLiveDashboard       src/utils/telegram_live.py
    Canli mac takibi. Pinlenen mesaj ile guncelleme.

  TelegramAdmin               src/utils/telegram_admin.py
    Yonetim paneli. Sistem kontrolu, config, backup.

  TelegramScenario            src/utils/telegram_scenario.py
    "What-if" senaryo simulatoru. Inline butonlarla.

  TelegramChartSender         src/ui/telegram_chart_sender.py
    Grafik gonderimi. Karanlik tema, matplotlib.

  AgentPollSystem             src/utils/agent_poll_system.py
    Coklu ajan oylama sistemi.

--- 9.2 RAPORLAMA ---

  DailyBriefing              src/utils/daily_briefing.py
    Gunluk brifing. LLM ozetleme. Her gun 09:00.

  StrategyHealthReport       src/utils/strategy_health_report.py
    PDF rapor. Sharpe Ratio, Drawdown, ROI, modul performansi.

  DecisionFlowGenerator      src/utils/decision_flow_gen.py
    Karar akisi gorsellestirme. Graphviz/Mermaid.

  WarRoom                    src/utils/war_room.py
    3 ajanli tartisma sistemi (Gambler, Risk Manager, Quant).
    LLM ile tartisma ve konsensus.

  DevilsAdvocate             src/utils/devils_advocate.py
    Karsi-analiz. "Bu bahis neden kaybedebilir?" raporu.
    Verdict: APPROVE / CAUTION / WARNING / REJECT

--- 9.3 ARAYUZLER ---

  StreamlitDashboard         src/ui/streamlit_dashboard.py
    Web dashboard. Plotly grafikleri. "python bahis.py web" ile acilir.
    Sayfalar: Dashboard, Value Finder, Mac Analizi, Kasa Egrisi,
              Canli Radar, Anomali Tespiti

  DashboardTUI               src/ui/dashboard_tui.py
    Terminal dashboard. Rich/Textual. Bloomberg tarzı.

  WebAppServer               src/ui/webapp_server.py
    FastAPI sunucu. Telegram Mini App entegrasyonu.
    Endpointler: /api/health, /api/portfolio, /api/signals,
                 /api/pnl-history, /api/config

--- 9.4 DIGER ARACLAR ---

  PodcastProducer            src/utils/podcast_producer.py
    Edge-TTS ile Turkce sesli rapor. MP3 cikti.

  PlotAnimator               src/utils/plot_animator.py
    Animasyonlu GIF uretimi. Heatmap, odds, basinc, bankroll.

  PsychoProfiler             src/utils/psycho_profiler.py
    Yatirimci psikoloji profili. Bias tespiti.

  QueryAssistant             src/utils/query_assistant.py
    Dogal dil veritabani sorgulama (Text-to-SQL).

  HumanFeedbackLoop          src/utils/human_feedback_loop.py
    RLHF. Insan kararlarini kaydetme ve odul hesaplama.

  HumanInTheLoop             src/ui/human_in_the_loop.py
    Insan kontrolu filtresi. Model vs insan performans karsilastirmasi.

  ThresholdController        src/utils/threshold_controller.py
    Dinamik esik deger ayarlama. EV, guven, max stake.

  AutoDocGenerator           src/utils/auto_doc_generator.py
    Otomatik API dokumantasyonu. AST parsing.

================================================================================
10. ANA ANALIZ DONGUSU (_analysis_loop)
================================================================================
!! Bu bolum _analysis_loop() fonksiyonu degistiginde guncellenmelidir !!

Her dongu iterasyonunda sirasıyla:

  ADIM 1: VERI TOPLAMA
    db.get_upcoming_matches(48)   -> yaklasan maclar
    validator.validate_batch()    -> veri dogrulama

  ADIM 2: OZELLIK MUHENDISLIGI
    cache.get_or_compute()        -> ozellik onbellekleme
    time_decay.apply()            -> zaman agirliklama
    jax_acc.accelerate()          -> GPU hizlandirma

  ADIM 3: MODEL TAHMINLERI (30+ model paralel)
    poisson, dixon_coles, monte_carlo, elo, gb, glm, bayesian,
    lstm, prob_engine, mtl, kan, kalman, vector_engine,
    digital_twin, quantum_brain, news_rag, sentiment, ve diger...

  ADIM 4: ISTATISTIKSEL ANALIZ
    evt_scanner, causal, conformal, path_sig, jump_diff,
    geometric, anomaly, nash, entropy, chaos, wavelet,
    topology, regime_switcher, survival, fatigue, ve diger...

  ADIM 5: ENSEMBLE BIRLESTIRME
    prob_engine.ensemble()        -> tum sinyalleri birlestir
    stacker.add_base_prediction() -> stacking

  ADIM 6: FISHER GEOMETRY
    fisher_geo.compare()          -> anomali ve rejim tespiti

  ADIM 7: EPISTEMIK FILTRE
    philo_engine.evaluate()       -> felsefi filtre

  ADIM 8: REJIM-KELLY STAKE
    regime_kelly.calculate()      -> rejim-farkinda stake

  ADIM 9: RISK FILTRELERI
    covar, bl_opt, risk_solver, pnl, fair_value, uncertainty,
    copula, entropy_filter, evt_risk, iso_anomaly, ve diger...

  ADIM 10: BILDIRIM VE KAYIT
    telegram -> bildirim gonder
    lance    -> vektor veritabanina kaydet
    neo4j    -> graf veritabanina kaydet
    db       -> sonuclari kaydet

  Her 5 dongude:
    cockpit.update_cockpit()  -> Telegram HUD guncelle

  Her 100 dongude:
    evolver.evolve()          -> strateji evrimi

================================================================================
11. ZAMANLANMIS GOREVLER (Scheduler)
================================================================================
!! Bu bolum scheduler gorevleri degistiginde guncellenmelidir !!

  Saat 02:00  log_rotator.rotate()     Eski loglari arsivle
  Saat 03:00  lstm.retrain()           LSTM modelini yeniden egit
  Saat 04:00  rl_agent.save()          RL checkpoint kaydet
  Saat 09:00  daily_briefing.send()    Gunluk brifing gonder
  Saat 10:00  data_sources.fetch_all() Toplu veri guncelle
  Her 5 dk    health.check()           Sistem saglik kontrolu
  Her 50 dng  automl.search()          AutoML model arama

================================================================================
12. YAPILANDIRMA DOSYALARI
================================================================================

  .env                    Gizli ortam degiskenleri
    TELEGRAM_BOT_TOKEN    Telegram bot tokeni
    TELEGRAM_CHAT_ID      Telegram sohbet ID
    NEO4J_URI             Neo4j baglanti adresi (opsiyonel)
    NEO4J_USER            Neo4j kullanici adi (opsiyonel)
    NEO4J_PASSWORD        Neo4j sifresi (opsiyonel)

  .gitignore              Git'ten haric tutulan dosyalar
  .cursorignore           AI context'ten haric tutulan dosyalar

  requirements.txt        Python bagimliliklari (444 paket)

  logs/                   Guncel loglar (son 3 gun)
    bot_YYYY-MM-DD.log    Gunluk bot logu
    error.log             Hata logu
  logs/archive/           Arsivlenmis loglar (kalici, silinmez)
    *.log.gz              Sikistirilmis arsiv

================================================================================
13. LOG YONETIMI
================================================================================

Log Seviyeleri:
  DEBUG   : Detayli bilgi (sadece dosyada)
  INFO    : Genel bilgi (konsol + dosya)
  WARNING : Uyarilar
  ERROR   : Hatalar (error.log'a da yazilir)

Log Dosyalari:
  logs/bot_YYYY-MM-DD.log  Gunluk log (max 10 MB, 3 gun sakla)
  logs/error.log           Sadece hatalar
  logs/archive/*.log.gz    Arsivlenmis eski loglar (kalici)

Log Rotasyonu:
  - 3 gunden eski loglar otomatik arsivlenir
  - Arsivlenen loglar gz olarak sikistirilir
  - Arsivlenen loglar ASLA silinmez
  - Her gece 02:00'da otomatik calisir
  - Boot sirasinda da bir kez calisir

================================================================================
14. HATA AYIKLAMA VE IZLEME
================================================================================

Exception Guardian:
  Error budget      : Saatte 50 hata (asildiginda alarm)
  Circuit threshold : 10 hata (modul devre disi)
  Circuit reset     : 300 saniye sonra yarim-acik dene
  Heartbeat         : 300 saniye sessizlik = uyari
  Siniflandirma     : network, data, model, system, unknown

Telemetry (OpenTelemetry):
  Service name      : "quant-betting-bot"
  Span izleme       : Her islem icin sure ve durum
  Bottleneck raporu : En yavas 10 islem

Prometheus:
  Port              : 9090
  Metrikler         : CPU, RAM, model latency, aktif gorevler

Doctor komutu:
  python bahis.py doctor
  Tum modulleri kontrol eder, eksikleri tespit eder.

================================================================================
15. BAGIMLILIKLAR VE TEKNOLOJILER
================================================================================

Temel:     numpy, scipy, polars, duckdb, pydantic, loguru
ML:        torch, scikit-learn, lightgbm, xgboost, transformers
Bayesian:  pymc, arviz, numpyro, statsmodels
Finans:    arch, vectorbt, PyPortfolioOpt, lifelines
Topoloji:  giotto-tda, ripser, kmapper
Zaman:     prophet, filterpy, hmmlearn
Quantum:   pennylane, tenseal
Scraping:  playwright, httpx, aiohttp, beautifulsoup4, lxml
Telegram:  python-telegram-bot
Dashboard: streamlit, plotly, fastapi, uvicorn, rich, textual
NLP:       vaderSentiment, textblob, sentencepiece
Vizyon:    opencv-python-headless, ultralytics (YOLO)
Ses:       edge-tts, openai-whisper
Dagitik:   ray, prefect, grpcio, msgpack
Graph:     neo4j, networkx, GraphRicciCurvature
Vektor:    lancedb, faiss-cpu
LLM:       google-generativeai, langchain, ollama
Diger:     numba, simanneal, diskcache, pyarrow, apscheduler

Toplam: 444 kurulu paket

Harici Programlar:
  Visual Studio Build Tools 2022  C/C++ derleyici
  CMake 4.2                       Build sistemi
  Rust 1.93 + Cargo               Rust uzantilari
  Visual C++ Redistributable      DLL runtime

================================================================================
16. GUNCELLEME KAYITLARI
================================================================================

  [2026-02-16] Ilk surum olusturuldu. Tum moduller belgelendi.

  Bir sonraki guncelleme icin:
  - Tarih ve degisiklik aciklamasini buraya ekleyin
  - Ilgili bolumu de guncelleyin
  - Ornek: [2026-03-01] Yeni modul: src/quant/yeni_modul.py (Bolum 7'ye eklendi)

================================================================================
  !! SON NOT !!
  Bu belge proje ile birlikte yasayan bir dokumantasyondur.
  Kodda yapilan her degisiklikte guncellenmelidir.
  Her bolumun basinda guncelleme gerektiren kosullar belirtilmistir.
================================================================================
