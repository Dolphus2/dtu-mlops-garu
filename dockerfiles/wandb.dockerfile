FROM ghcr.io/astral-sh/uv:python3.12-bookworm
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY wandb_tester.py wandb_tester.py
ENTRYPOINT ["uv", "run", "wandb_tester.py"]