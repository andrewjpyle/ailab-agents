# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12-slim

# ---- builder: resolve the locked environment into /opt/venv with uv ----------
FROM ${PYTHON_IMAGE} AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /usr/local/bin/uv

# git is needed by uv to fetch the ailab-core git dependency during the build.
# Builder-stage only: the runtime stage copies the resolved /opt/venv and needs no git.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Dependencies first so this layer is cached across source-only changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- runtime: slim interpreter + venv, no build tools, non-root -------------
FROM ${PYTHON_IMAGE} AS runtime

ARG GIT_COMMIT=unknown

RUN groupadd --system --gid 10001 ailab \
 && useradd --system --uid 10001 --gid ailab --no-create-home --shell /usr/sbin/nologin ailab

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY eval_config.toml ./
COPY fixtures ./fixtures
RUN mkdir -p /app/results && chown ailab:ailab /app/results

# 12-factor: every runtime setting is an environment variable with a sane default.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AILAB_COMMIT=${GIT_COMMIT} \
    AILAB_CONFIG=/app/eval_config.toml \
    AILAB_OUTPUT=/app/results/eval_results.json

USER ailab:ailab

CMD ["ailab-demo"]
