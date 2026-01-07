FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base
# FROM python:3.12-slim

# Small official pytorch image
# FROM pytorch/pytorch:2.6.0-cuda13.0-cudnn9-runtime

# Big official nvidia image with lots of libraries
# FROM nvcr.io/nvidia/pytorch:25.08-py3


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

ENTRYPOINT ["uv", "run", "src/dtu_mlops_garu/train.py"]
