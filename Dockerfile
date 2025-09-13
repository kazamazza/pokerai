# API-only image (no solver, no worker stage)
FROM python:3.11-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Optional system deps (uncomment if you need to compile wheels)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Install runtime deps
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir -r /app/requirements-api.txt

# Copy application code
COPY . /app

# Run as non-root
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# (Optional) healthcheck if curl is available in your requirements
# HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
#   CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]