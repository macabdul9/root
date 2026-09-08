# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.10 AS uv
FROM python:3.12-slim-trixie

# uv.lock supplies PyTorch and its CUDA runtime libraries. The NVIDIA driver
# comes from the host through Docker --gpus or Apptainer --nv.

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    HOME=/tmp

WORKDIR /opt/root
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/tmp/.cache/uv \
    uv sync --locked --extra dev --no-install-project --no-editable
COPY src ./src
COPY tests ./tests
RUN --mount=type=cache,target=/tmp/.cache/uv \
    uv sync --locked --extra dev --no-editable
COPY containers/test.sh /usr/local/bin/root-container-test
RUN chmod 755 /usr/local/bin/root-container-test \
    && useradd --uid 1000 --no-create-home rootapp \
    && mkdir -p /workspace /cache/huggingface \
    && chmod 1777 /workspace /cache /cache/huggingface

USER rootapp
RUN root --version && root-container-test
WORKDIR /workspace
CMD ["root", "--engine", "transformers", "--device", "cuda"]
