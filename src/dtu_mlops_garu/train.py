from pathlib import Path
import typer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dtu_mlops_garu.data import PROCESSED_DATA_PATH, RAW_DATA_PATH, corrupt_mnist, preprocess_mnist
from dtu_mlops_garu.model import Model1, Model2
from dtu_mlops_garu.utils import train_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

train_app = typer.Typer(help="Train commands")

@train_app.command()
def train(model_path: str = "models/trained_model.pth", lr: float = 1e-3, batch_size: int = 64, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")  # OOOOH Cool!
    print(f"Using {DEVICE} device")
    print(torch.cuda.is_available())

    if not (PROCESSED_DATA_PATH / "train_images.pt").exists():
        preprocess_mnist(RAW_DATA_PATH, PROCESSED_DATA_PATH)
        assert (PROCESSED_DATA_PATH / "train_images.pt").exists()

    model = Model2().to(DEVICE)
    training_data, _ = corrupt_mnist(PROCESSED_DATA_PATH)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    statistics = {"train_loss": [], "train_accuracy": []}
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_statistics = train_utils.train_epoch(train_dataloader, model, criterion, optimizer, DEVICE)
        statistics["train_accuracy"] += epoch_statistics["train_accuracy"]
        statistics["train_loss"] += epoch_statistics["train_loss"]

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved PyTorch Model State to {model_path}")

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

