# Northwind Freight - application image
#
# Constraints this image is built under, because it runs inside the customer's
# VPC and not in your account:
#   - no secrets baked in, ever (they arrive at runtime from the customer's
#     secret store, so a leaked image layer is not a leaked credential)
#   - non-root, because the customer's container policy will reject root
#   - multi-stage, so build tooling does not ship to production
#   - a healthcheck, because their orchestrator decides restarts, not you

# --- Stage 1: build -------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build-only dependencies. These never reach the runtime image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency manifest alone first so the layer caches across code edits.
COPY requirements.txt .

# Wheels are built here and installed in the runtime stage. Installing into a
# prefix keeps the copy to a single predictable directory.
RUN python -m pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt


# --- Stage 2: runtime -----------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/usr/local/bin:$PATH

# Non-root, fixed uid/gid so volume permissions are predictable across hosts.
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /install /usr/local

# Only what the service needs at runtime. tests/, docs/, and eval/ are excluded
# deliberately: the golden set contains customer ticket text and does not belong
# in an image that may be pushed to a shared registry.
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser deploy/jwt_middleware.py ./deploy/jwt_middleware.py
COPY --chown=appuser:appuser configs/ ./configs/

USER appuser

EXPOSE 8080

# NO secrets here. Not as ENV, not as ARG, not as a COPY of .env.
# ARG values persist in image history and are readable by anyone with the image.
# Runtime configuration arrives from the customer's secret manager via the
# orchestrator. The variables below are non-secret defaults only.
ENV APP_PORT=8080 \
    FDE_LOG_LEVEL=INFO

# The orchestrator restarts on this. Keep it cheap and dependency-free: a
# healthcheck that calls the model is a healthcheck that bills you to be alive.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status==200 else 1)" || exit 1

CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
