import matplotlib.pyplot as plt
import torch
import typer
from dtu_mlops_garu.data import corrupt_mnist, preprocess_mnist, RAW_DATA_PATH, PROCESSED_DATA_PATH
from dtu_mlops_garu.model import Model
from dtu_mlops_garu.utils import train_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(lr: float = 1e-3, batch_size: int = 64, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}") # OOOOH Cool!
    print(f"Using {DEVICE} device")

    if not (PROCESSED_DATA_PATH / "train_images.pt").exists():
        preprocess_mnist(RAW_DATA_PATH, PROCESSED_DATA_PATH)
        assert (PROCESSED_DATA_PATH / "train_images.pt").exists()

    model = Model(c1 = 128).to(DEVICE)
    training_data, _ = corrupt_mnist(PROCESSED_DATA_PATH)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    criterion  = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    statistics = {"train_loss": [], "train_accuracy": []}
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_statistics = train_utils.train_epoch(train_dataloader, model, criterion, optimizer, DEVICE)
        statistics["train_accuracy"] += epoch_statistics["train_accuracy"]
        statistics["train_loss"] += epoch_statistics["train_loss"]

    torch.save(model.state_dict(), "models/model.pth")
    print("Saved PyTorch Model State to models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
