# 🚀 Quant Betting Bot - Tam Kurulum Rehberi

Sistemin "Tam Potansiyel" (Singularity/Sovereign) seviyesinde çalışması için gerekli tüm bileşenler aşağıdadır.

## 1. Sistem Gereksinimleri
*   **İşletim Sistemi**: Windows 10/11 (PowerShell/CMD)
*   **Python**: 3.10 veya 3.11 (Önerilen)
*   **Disk**: ~10GB (Modeller ve DB'ler için)
*   **RAM**: Minimum 8GB (Ollama için 16GB önerilir)

## 2. Zorunlu Programlar (Dış Bağımlılıklar)

### A. Ollama (Yerel Yapay Zeka)
Botun "Sokratik Tartışma", "Red Team" ve "NLP" özelliklerini yerel olarak çalıştırması için:
1.  [ollama.com](https://ollama.com/) adresinden indirin ve kurun.
2.  Terminali açıp şu modelleri çekin:
    ```bash
    ollama pull llama3
    ollama pull phi3
    ```

### B. Playwright (Web Scraper)
Flashscore ve diğer kaynaklardan veri çekmek için:
```bash
pip install playwright
playwright install chromium
```

## 3. Python Kütüphaneleri (Tam Liste)

Aşağıdaki komutu kopyalayıp terminale yapıştırarak tüm potansiyeli aktif edebilirsiniz:

```bash
# Temel & Veri Yönetimi
pip install polars duckdb loguru numba psutil httpx pydantic

# Analiz & İstatistik
pip install scipy numpy ta lancedb PyMC

# Görselleştirme (Telegram Dashboard)
pip install matplotlib plotly

# Kullanıcı Arayüzü & İletişim
pip install python-telegram-bot

# Otonom & Ses (Phase 15)
pip install openai-whisper setuptools-rust
```

## 4. Opsiyonel: Rust Math Engine
Hesaplama hızını 100 kat artırmak isterseniz:
1.  [Rustup](https://rustup.rs/) kurun.
2.  `src/core/rust_engine` dizininde derleme yapın (Sistem derlenmiş halini de arar).

## 5. Başlatma
Hepsini kurduktan sonra botu başlatmak için:
```bash
python bahis.py
```

---
**Not**: Tüm bu araçlar tamamen **ÜCRETSİZDİR**. Herhangi bir API key (The Odds API hariç, onun da ücretsiz kotası vardır) gerektirmez.
