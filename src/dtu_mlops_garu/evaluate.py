from pathlib import Path

import torch

from dtu_mlops_garu.data import PROCESSED_DATA_PATH, corrupt_mnist
from dtu_mlops_garu.model import Model1
from dtu_mlops_garu.utils import train_utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = Model1(c1=128).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist(PROCESSED_DATA_PATH)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64)

    criterion = torch.nn.CrossEntropyLoss()
    train_utils.test(test_dataloader, model, criterion, DEVICE)
    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    model_checkpoint = Path("models/model.pth")
    evaluate(model_checkpoint)
