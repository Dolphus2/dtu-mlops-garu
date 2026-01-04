# FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

COPY src/ src/
COPY data/ data/

COPY dockerfiles/eval_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Install dependencies only (persists across Docker builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# ENTRYPOINT ["uv", "run", "src/dtu_mlops_garu/evaluate.py", "models/trained_model.pth"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# optional: default args (can be overridden by `docker run`)
CMD []