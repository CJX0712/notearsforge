# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install build-time dev tools first (kept out of the final runtime image layer
# is unnecessary here; this is a CPU-only, zero-weight system).
COPY requirements.lock.txt requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock.txt \
    && pip install --no-cache-dir "pytest>=8" "ruff==0.16.10" pytest-cov

COPY . .

# Build-time hard gate: lint + format check + tests. No `|| true` -- a red lint
# must fail the build, exactly as CI does.
RUN python -m ruff check . \
    && python -m ruff format --check . \
    && python -m pytest -q -W ignore::UserWarning

# Smoke + determinism check (exit non-zero on regression).
RUN python examples/run_demo.py

ENTRYPOINT ["python", "cli.py"]
CMD ["run"]
