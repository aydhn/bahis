# 🤖 Quant Betting Bot

**Otonom Çoklu Ajan Hedge Fund Mimarisine Sahip Bahis Analiz ve Karar Destek Sistemi**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 Genel Bakış

Quant Betting Bot, spor bahisleri için geliştirilmiş, kantitatif analiz, makine öğrenmesi ve yapay zeka tekniklerini kullanan gelişmiş bir analiz sistemidir. Geleneksel tahmin botlarından farklı olarak, bir hedge fund gibi çalışır; çok katmanlı risk yönetimi, portföy optimizasyonu ve otonom karar mekanizmaları içerir.

Sistem, veri toplamadan (ingestion) karmaşık kantitatif modellere (quant), risk yönetiminden (risk) raporlamaya kadar uzanan modüler bir mimariye sahiptir. Yaklaşık 150 modül ve 400+ kütüphane kullanılarak inşa edilmiştir.

## 🚀 Özellikler

Sistem 6 ana katmandan oluşur:

*   **Katman 0 - Altyapı:** Devre kesici (Circuit Breaker), akıllı önbellek (Smart Cache), yapılandırılmış loglama ve otomatik hata iyileştirme (Self-Healing).
*   **Katman 1 - Veri Toplama:** Web scraping, API entegrasyonları, gerçek zamanlı görüntü işleme (Vision) ve haber analizi.
*   **Katman 2 - Hafıza:** Polars tabanlı veri işleme, vektör veritabanı (LanceDB), graf veritabanı (Neo4j) ve anlamsal hafıza.
*   **Katman 3 - Kantitatif Zeka:** 30+ İstatistiksel ve ML Modeli:
    *   Poisson, Dixon-Coles, Monte Carlo
    *   Gradient Boosting (LightGBM/XGBoost), LSTM, Transformer
    *   Bayesian Hiyerarşik Modeller, Kuantum ML, Kaos Teorisi
*   **Katman 4 - Risk ve İcra:** Kelly Kriteri, Markowitz Portföy Optimizasyonu, Hedge stratejileri ve Shadow Trading.
*   **Katman 5 - Arayüz:** Telegram botu, Streamlit dashboard, sesli asistan ve otomatik raporlama.

## 🛠️ Teknolojiler

Proje, modern ve yüksek performanslı kütüphaneler üzerine kurulmuştur:

*   **Veri & Hesaplama:** `numpy`, `scipy`, `polars`, `duckdb`
*   **Makine Öğrenmesi:** `torch`, `scikit-learn`, `lightgbm`, `xgboost`
*   **Bayesian & İstatistik:** `pymc`, `arviz`, `statsmodels`
*   **Finans:** `vectorbt`, `PyPortfolioOpt`
*   **Scraping:** `playwright`, `httpx`, `beautifulsoup4`
*   **Sistem:** `ray`, `prefect`, `typer`, `loguru`
*   **Dashboard:** `streamlit`, `rich`, `fastapi`

## 📦 Kurulum

### Gereksinimler
*   Python 3.11+
*   Visual Studio Build Tools 2022 (C++ derleyici)
*   CMake 4.x
*   Rust 1.93+ (Cargo dahil)
*   Git

### Adım Adım Kurulum

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/username/quant-betting-bot.git
    cd quant-betting-bot
    ```

2.  **Sanal Ortam Oluşturun:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Bağımlılıkları Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Yapılandırma:**
    `.env.example` dosyasını `.env` olarak kopyalayın ve gerekli anahtarları girin:
    ```env
    TELEGRAM_BOT_TOKEN=your_token_here
    TELEGRAM_CHAT_ID=your_chat_id
    # Opsiyonel: Neo4j, OpenAI vb.
    ```

## 💻 Kullanım

Proje, `bahis.py` üzerinden yönetilen güçlü bir CLI (Komut Satırı Arayüzü) sunar.

### Temel Komutlar

*   **Botu Başlatma:**
    ```bash
    python bahis.py run --mode full
    ```
    Tüm katmanları (veri toplama, analiz, Telegram botu) ayağa kaldırır.

*   **Tek Maç Analizi:**
    ```bash
    python bahis.py analyze "Galatasaray" "Fenerbahçe"
    ```
    Belirtilen maç için detaylı kantitatif analiz raporu üretir.

*   **Sistem Sağlık Kontrolü:**
    ```bash
    python bahis.py doctor
    ```
    Eksik bağımlılıkları ve sistem durumunu kontrol eder.

*   **Web Dashboard:**
    ```bash
    python bahis.py web
    ```
    Streamlit arayüzünü tarayıcıda açar.

### Diğer Analiz Araçları

*   `python bahis.py nash <ev> <dep>`: Oyun teorisi (Nash Dengesi) analizi.
*   `python bahis.py chaos <ev> <dep>`: Kaos teorisi ve Lyapunov üssü analizi.
*   `python bahis.py simulation <ev> <dep>`: Monte Carlo simülasyonu.
*   `python bahis.py what-if <ev> <dep>`: "Ya şöyle olursa?" senaryo analizleri.

## 🏗️ Mimari Detaylar

Sistem, **_analysis_loop** fonksiyonu içerisinde sürekli bir döngüde çalışır:
1.  **Veri Toplama:** Yaklaşan maçlar çekilir ve doğrulanır.
2.  **Özellik Mühendisliği:** Veriler işlenir ve özellik matrisleri oluşturulur.
3.  **Tahmin:** 30'dan fazla model paralel olarak çalıştırılır.
4.  **İstatistiksel Analiz:** Kaos, entropi ve topolojik analizler yapılır.
5.  **Ensemble:** Farklı modellerin çıktıları birleştirilir.
6.  **Risk Yönetimi:** Portföy optimizasyonu ve Kelly kriteri ile bahis miktarları belirlenir.
7.  **Karar:** Nihai karar (Bahis Yap / Pas Geç) verilir ve Telegram üzerinden bildirilir.

## ⚠️ Yasal Uyarı

Bu yazılım **sadece eğitim ve araştırma amaçlıdır**. Gerçek para ile bahis oynamak risk içerir ve finansal kayıplara yol açabilir. Yazılımın geliştiricileri, bu botun kullanımından doğabilecek herhangi bir kayıptan sorumlu tutulamaz. Bahis oynamak, bulunduğunuz ülkenin yasalarına tabi olabilir. Lütfen sorumlu bahis oynayın.

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen önce bir issue açarak yapmak istediğiniz değişikliği tartışın.

1.  Forklayın.
2.  Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`).
3.  Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik: X'`).
4.  Branch'inizi pushlayın (`git push origin feature/yeni-ozellik`).
5.  Pull Request oluşturun.

---
*Dokümantasyon Tarihi: 2026-02-22*
