# ==============================================================================
# Backend Service Dockerfile (Multi-stage, Minimalist Base)
# ==============================================================================

# ── Stage 1: Builder ───────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# Install bare minimum compilation utilities for any source-distribution wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with no cache to keep layer footprint minimal
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runner ───────────────────────────────────────────────────────────
FROM python:3.13-slim AS runner

WORKDIR /app

# Install only bare minimum runtime shared libraries (libgomp for OpenMP, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Security: Create and run under an unprivileged non-root user
RUN useradd --create-home --uid 10001 --shell /bin/bash appuser \
    && mkdir -p /app/data /app/uploads \
    && chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser . /app

USER appuser

# Expose backend REST API port
EXPOSE 8000

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/stats || exit 1

# Launch production server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

