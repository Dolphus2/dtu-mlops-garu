import logging
from pathlib import Path

import torch
import typer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from dtu_mlops_garu.data import PROCESSED_DATA_PATH, get_corrupt_mnist
from dtu_mlops_garu.model import Model2
from dtu_mlops_garu.utils import train_utils
from dtu_mlops_garu.utils.load_utils import find_model_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

log = logging.getLogger(__name__)
evaluate_app = typer.Typer(help="Evaluate commands")


@evaluate_app.command()
def evaluate(
    model_checkpoint: str,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> None:
    """Evaluate a trained model."""
    log.info("Evaluating like my life depends on it")
    log.info(f"{model_checkpoint=}")
    model_checkpoint = find_model_path(model_checkpoint)
    log.info(f"{Path.cwd()=}")

    # Initialize Hydra and load config
    config_dir = "../../config"
    with initialize(version_base=None, config_path=config_dir):
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides or [])
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    batch_size = cfg.get("batch_size", 64)

    model = Model2().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))

    _, test_set = get_corrupt_mnist(PROCESSED_DATA_PATH)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    train_utils.test(test_dataloader, model, criterion, DEVICE)
    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    log.info(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    evaluate_app()
    # model_checkpoint = Path("models/model.pth")
    # evaluate(model_checkpoint)
