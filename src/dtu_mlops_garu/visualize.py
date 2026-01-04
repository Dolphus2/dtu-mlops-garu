import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dtu_mlops_garu.data import PROCESSED_DATA_PATH, corrupt_mnist
from dtu_mlops_garu.model import Model2
from dtu_mlops_garu.utils.load_utils import find_model_path

DEVICE = torch.device("cpu")

log = logging.getLogger(__name__)
visualize_app = typer.Typer(help="visualize commands")


@visualize_app.command()
def visualize(
    model_checkpoint: str,
    figure_name: str = "embeddings.png",
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> None:
    log.info("Visualizing like my life depends on it")
    log.info(f"Model checkpoint: {model_checkpoint}")
    model_checkpoint = find_model_path(model_checkpoint)

    # Initialize Hydra and load config
    config_dir = "../../config"
    with initialize(version_base=None, config_path=config_dir):
        cfg: DictConfig = compose(config_name=config_name, overrides=overrides or [])
    log.info(f"configuration: \n {OmegaConf.to_yaml(cfg)}")

    batch_size = cfg.get("batch_size", 64)

    model = Model2().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint, map_location=DEVICE))
    model.eval()
    # model.classifier = torch.nn.Identity

    _, test_set = corrupt_mnist(PROCESSED_DATA_PATH)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in test_dataloader:
            images, target = batch
            predictions = model.extract_features(images).flatten(start_dim=1)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    log.info(f"{embeddings.shape}")

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    reports_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    visualize_app()
    # typer.run(visualize)
    # model_checkpoint = Path("models/trained_model.pth")
    # visualize(model_checkpoint)
