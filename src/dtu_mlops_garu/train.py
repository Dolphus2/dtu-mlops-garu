import logging
import time
from pathlib import Path

import matplotlib
import numpy as np
import typer
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dtu_mlops_garu.data import PROCESSED_DATA_PATH, RAW_DATA_PATH, corrupt_mnist, preprocess_mnist
from dtu_mlops_garu.utils import train_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
train_app = typer.Typer(help="Train commands")


@train_app.command()
def train(
    model_path: str = "models/trained_model.pth",
    config_name: str = "train_config",
    overrides: list[str] = typer.Option(None, "--overrides"),
) -> None:
    """Train a model on MNIST."""
    log.info("Training day and night")

    # Initialize Hydra and load config
    config_dir = "../../config"
    with initialize(version_base=None, config_path=config_dir):
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides or [])
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    run = wandb.init(
        project="dtu_mlops_garu",
        name=f"dtu_mlops_garu_corrupt_mnist_{int(time.time())}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Use config values
    torch.manual_seed(cfg.get("seed", 42))
    batch_size = cfg.get("batch_size", 64)
    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 1e-3)
    momentum = cfg.get("momentum", 0.9)

    log.info(f"{lr=}, {batch_size=}, {epochs=}")  # OOOOH Cool!
    log.info(f"Using {DEVICE} device")
    log.info(f"{torch.cuda.is_available()}")

    if not (PROCESSED_DATA_PATH / "train_images.pt").exists():
        preprocess_mnist(RAW_DATA_PATH, PROCESSED_DATA_PATH)
        assert (PROCESSED_DATA_PATH / "train_images.pt").exists()

    # Instantiate model from config using Hydra's instantiate
    model = instantiate(cfg.model).to(DEVICE)  # Very cool
    log.info(f"Loaded model: {cfg.model._target_}")

    training_data, _ = corrupt_mnist(PROCESSED_DATA_PATH)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    stats = {"train_loss": [], "train_accuracy": [], "precision": [], "recall": [], "f1": []}
    for t in range(epochs):
        log.info(f"Epoch {t + 1}\n-------------------------------")
        epoch_stats = train_utils.train_epoch(train_dataloader, model, criterion, optimizer, DEVICE)
        stats["train_accuracy"].append(epoch_stats["train_accuracy"])
        stats["train_loss"].append(epoch_stats["train_loss"])
        stats["precision"].append(epoch_stats["precision"])
        stats["recall"].append(epoch_stats["recall"])
        stats["f1"].append(epoch_stats["f1"])

        avg_loss = np.mean(epoch_stats["train_loss"])
        avg_acc = np.mean(epoch_stats["train_accuracy"])
        wandb.log(
            {
                "epoch": t + 1,
                "train_loss": avg_loss,
                "train_accuracy": avg_acc,
                "precision": epoch_stats["precision"],
                "recall": epoch_stats["recall"],
                "f1": epoch_stats["f1"],
            }
        )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    log.info(f"Saved PyTorch Model State to {model_path}")
    wandb.log_model(path=model_path, name="corruptmnist_model")
    log.info("Logged model to wandb")

    # first we save the model to a file then log it as an artifact
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={
            "accuracy": stats["train_accuracy"][-1],
            "precision": stats["precision"][-1],
            "recall": stats["recall"][-1],
            "f1": stats["f1"][-1],
        },
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    reports_dir = Path("reports") / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports_file = reports_dir / "training_statistics.png"
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(stats["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(stats["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(reports_file)
    wandb.log({"training_plots": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    train_app()
