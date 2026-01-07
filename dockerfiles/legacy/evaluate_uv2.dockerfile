FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder
WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

COPY src/ src/
COPY data/ data/

# Install dependencies only (persists across Docker builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

FROM nvcr.io/nvidia/pytorch:25.08-py3 AS runtime
WORKDIR /app
COPY --from=builder /.venv /.venv
ENV PATH="/.venv/bin:${PATH}"
COPY src/ ./src

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# optional: default args (can be overridden by `docker run`)
CMD []
