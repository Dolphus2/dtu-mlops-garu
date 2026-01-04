import os
from pathlib import Path
from invoke import Context, task, Collection

WINDOWS = os.name == "nt"
PROJECT_NAME = "dtu_mlops_garu"
PYTHON_VERSION = "3.12"

@task
def python(ctx: Context):
    """ """
    ctx.run(
        command = "which python" if os.name != "nt" else "where python",
        pty = not WINDOWS,
    )

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/data.py data/raw/corruptmnist data/processed/corruptmnist",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context, model_path: str = "models/trained_model.pth") -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py --model-path {model_path}", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, model_checkpoint: str) -> None:
    """Evaluate model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py {model_checkpoint}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "auto") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t evaluate:latest . -f dockerfiles/evaluate.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    # ctx.run(
    #     f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    # )

@task
def docker_run_train(ctx: Context, model_path: str = "models/trained_model.pth") -> None:
    """Run the train image with an optional model path mounted."""
    ctx.run(
        f"docker run --name train --rm -v {Path.cwd()}/models/{Path(model_path).name}:/app/models/{Path(model_path).name} train:latest --model-path {model_path}",
        echo=True,
        pty=not WINDOWS,
    )

@task
def docker_run_evaluate(ctx: Context) -> None:
    """Run the evaluate image using a mounted model"""
    host = Path.cwd()
    ctx.run(
        f"docker run --name evaluate --rm \
            -v {host}/models/trained_model.pth:/app/models/trained_model.pth \
            -v {host}/data/test_images.pt:/app/test_images.pt \
            -v {host}/data/test_targets.pt:/app/test_targets.pt \
            -v {host}/reports/figures:/app/reports/figures\
            evaluate:latest \
            models/trained_model.pth",
        echo=True,
        pty=not WINDOWS,
    )

# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
