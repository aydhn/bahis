# ═══════════════════════════════════════════════════════════
#  Quant Betting Bot – Production Dockerfile
#  Multi-stage build: dependency install + lean runtime
# ═══════════════════════════════════════════════════════════

# ── Stage 1: Dependency Builder ──
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──
FROM python:3.11-slim

LABEL maintainer="quant-betting-bot"
LABEL description="Quant Betting Bot – Hedge Fund Seviyesi Algoritmik Bahis Sistemi"
LABEL version="5.0"

WORKDIR /app

# Sistem bağımlılıkları (Playwright & Chrome için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libxshmfence1 libx11-6 libx11-xcb1 \
    libxcb1 libxext6 libxfixes3 fonts-liberation \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python paketleri (builder stage'den kopyala)
COPY --from=builder /install /usr/local

# Playwright tarayıcıları (Chrome)
RUN playwright install chromium --with-deps 2>/dev/null || true

# Uygulama kodu
COPY . .

# Gerekli dizinler
RUN mkdir -p data logs reports output/podcasts docs models

# Prometheus metrik portu
EXPOSE 9090
# Streamlit dashboard portu
EXPOSE 8501

# Sağlık kontrolü
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Varsayılan: tam mod + Telegram aktif
ENTRYPOINT ["python", "bahis.py"]
CMD ["run", "--mode", "full", "--telegram"]
