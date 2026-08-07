# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for the KMS Core AI RAG Engine.
# Follows the same hackathon starter-kit conventions used by the orchestrator:
# python 3.12, uv package manager, multi-stage build, and a non-root runtime user.

# ------------------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Install uv for fast, reproducible Python dependency management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency definitions first to maximise Docker layer caching.
COPY pyproject.toml uv.lock ./

# Create the virtual environment at /app/.venv and install production deps.
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -r pyproject.toml

# ------------------------------------------------------------------------------
# Stage 2: Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Create a non-root user for production runtime security.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home --shell /bin/bash appuser

# Install gosu so the entrypoint can drop from root to appuser at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gosu && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the prepared virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Make the virtual environment binaries available on PATH.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application source code.
COPY src/ ./src/

# Copy helper scripts (e.g., privilege-dropping entrypoint).
COPY scripts/ ./scripts/

# Runtime directories for vector DB, logs, and evaluator outputs.
RUN mkdir -p /app/data /app/logs /app/outputs && \
    chown -R appuser:appuser /app && \
    chmod +x /app/scripts/entrypoint.sh

# The container starts as root so the entrypoint can fix bind-mount ownership,
# then it drops to appuser before running the application.
USER root

# Expose the Core AI service port.
EXPOSE 8001

# Health check aligned with the starter-kit / compose probe (30s interval).
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/v1/health')" || exit 1

# Use the entrypoint to drop privileges after fixing runtime permissions.
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

# Launch Uvicorn via the Python module runner to avoid shebang mismatches
# between multi-stage build layers. --app-dir src adds src/ to PYTHONPATH so
# relative imports such as `from pipelines...` resolve correctly.
CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8001"]
