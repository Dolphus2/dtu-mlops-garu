# FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

COPY dockerfiles/eval_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# optional: default args (can be overridden by `docker run`)
CMD []
