FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base
# FROM python:3.12-slim

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# Cache uv downloads and builds (persists across Docker builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src/ data/ ./

# Cache mount again for the final sync (installing your project)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

WORKDIR /app

ENTRYPOINT ["uv", "run", "src/dtu_mlops_garu/train.py"]
