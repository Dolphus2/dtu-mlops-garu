from pathlib import Path
import typer
import logging
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from hydra.utils import instantiate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dtu_mlops_garu.data import PROCESSED_DATA_PATH, RAW_DATA_PATH, corrupt_mnist, preprocess_mnist
from dtu_mlops_garu.model import Model1, Model2
from dtu_mlops_garu.utils import train_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

train_app = typer.Typer(help="Train commands")

@train_app.command()
def train(
    model_path: str = "models/trained_model.pth", 
    config_name: str = "train_config",
    overrides: list[str] | None = None,
) -> None:
    """Train a model on MNIST."""
    log.info("Training day and night")

    # Initialize Hydra and load config
    config_dir = "../../config"
    with initialize(version_base=None, config_path=config_dir):
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides or [])
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

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
    model = instantiate(cfg.model).to(DEVICE) # Very cool
    log.info(f"Loaded model: {cfg.model._target_}")

    training_data, _ = corrupt_mnist(PROCESSED_DATA_PATH)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    statistics = {"train_loss": [], "train_accuracy": []}
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_statistics = train_utils.train_epoch(train_dataloader, model, criterion, optimizer, DEVICE)
        statistics["train_accuracy"] += epoch_statistics["train_accuracy"]
        statistics["train_loss"] += epoch_statistics["train_loss"]

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    log.info(f"Saved PyTorch Model State to {model_path}")

    reports_dir = Path("reports") / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports_file = reports_dir / "training_statistics.png"
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(reports_file)
    plt.close(fig)

if __name__ == "__main__":
    train_app() 

