import logging
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt  # only needed for plotting
import torch
import typer
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting
from torch.utils.data import DataLoader, Dataset

N_TRAIN_FILES = 5

RAW_DATA_PATH = Path("data") / "raw" / "corruptmnist"
PROCESSED_DATA_PATH = Path("data") / "processed" / "corruptmnist"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
data_app = typer.Typer(help="Data commands")


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""


def _download_corrupt_mnist(output_dir: Path = RAW_DATA_PATH) -> None:
    """
    Download the CorruptMNIST dataset from Google Drive using gdown.

    Parameters
    ----------
    output_dir : Path
        Directory where the raw dataset will be downloaded. Defaults to RAW_DATA_PATH.

    """
    try:
        subprocess.run(["uv", "add", "gdown"], check=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "uv",
                "run",
                "gdown",
                "--folder",
                "https://drive.google.com/drive/folders/1ddWeCcsfmelqxF8sOGBihY9IU98S9JRP?usp=sharing",
                "-O",
                str(output_dir),
            ],
            check=True,
        )
        log.info(f"Dataset downloaded successfully to {output_dir}")
    except subprocess.CalledProcessError as e:
        log.error(f"Error downloading dataset: {e}")
        raise


def prepare_corrupt_mnist(raw_data_path: Path = RAW_DATA_PATH, processed_data_path: Path = PROCESSED_DATA_PATH) -> None:
    """Prepare CorruptMNIST dataset by downloading and preprocessing if needed."""
    if not (processed_data_path / "train_images.pt").is_file():
        if not (raw_data_path / "train_images_0.pt").is_file():
            _download_corrupt_mnist(raw_data_path)
        preprocess_mnist(raw_data_path, processed_data_path)
    else:
        log.info("Processed dataset already present")


@data_app.command()
def get_corrupt_mnist(
    data_path: Path = PROCESSED_DATA_PATH, raw_data_path: Path = RAW_DATA_PATH
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load processed CorruptMNIST tensors from disk and return PyTorch datasets.

    Parameters
    ----------
    data_path : Path
        Directory containing the processed tensor files. Expected files:
        - train_images.pt : Tensor of training images, shape (N_train, C, H, W) or (N_train, H, W)
        - train_target.pt  : Tensor of training labels, shape (N_train,)
        - test_images.pt  : Tensor of test images, shape (N_test, C, H, W) or (N_test, H, W)
        - test_target.pt   : Tensor of test labels, shape (N_test,)

    Returns
    -------
    tuple[TensorDataset, TensorDataset]
        (train_set, test_set) where each is a torch.utils.data.TensorDataset that
        yields (image, label) pairs suitable for use with torch DataLoader.

    """
    if not data_path.exists() or not (data_path / "train_images.pt").is_file():
        if not raw_data_path.exists() or not (raw_data_path / "train_images_0.pt").is_file():
            _download_corrupt_mnist(raw_data_path)
            assert raw_data_path.exists() and (raw_data_path / "train_images_0.pt").is_file()
        preprocess_mnist(raw_data_path, data_path)
    assert data_path.exists() and (data_path / "train_images.pt").is_file()

    train_images: torch.tensor = torch.load(f"{data_path}/train_images.pt", weights_only=True)
    train_target: torch.tensor = torch.load(f"{data_path}/train_target.pt", weights_only=True)

    test_images: torch.Tensor = torch.load(f"{data_path}/test_images.pt", weights_only=True)
    test_target: torch.Tensor = torch.load(f"{data_path}/test_target.pt", weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


def get_dataloaders(
    data_path: Path, batch_size: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    training_data, test_set = get_corrupt_mnist(data_path)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_dataloader, test_dataloader


def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, label in zip(grid, images, target):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


def normalize(images: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(images.ndim))  # Wrong I guess. Should be over all images
    train_normalized = images - images.mean(dims, keepdim=True)
    train_normalized /= train_normalized.std(dims, keepdim=True)
    return train_normalized


@data_app.command()
def preprocess_mnist(src: Path, dst: Path) -> None:
    """Preprocess data from src to DST."""
    log.info("Preprocessing data...")
    train_images, train_target = [], []
    for i in range(N_TRAIN_FILES + 1):
        train_images.append(torch.load(f"{src}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{src}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{src}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{src}/test_target.pt")

    train_images = train_images.unsqueeze(1).float()  ## add an extra dimension
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    dst.mkdir(exist_ok=True, parents=True)
    torch.save(normalize(train_images), f"{dst}/train_images.pt")
    torch.save(train_target, f"{dst}/train_target.pt")
    torch.save(normalize(test_images), f"{dst}/test_images.pt")
    torch.save(test_target, f"{dst}/test_target.pt")
    typer.echo(f"preprocess {src} -> {dst}")


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    # typer.run(prepare_corrupt_mnist)
    # preprocess(data_path, output_folder)
    train_set, test_set = get_corrupt_mnist(PROCESSED_DATA_PATH)
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    print(f"Mean of training data: {train_set.tensors[0].mean()}")
    print(f"Std of training data: {train_set.tensors[0].std()}")
    print(f"Mean of test data: {test_set.tensors[0].mean()}")
    print(f"Std of test data: {test_set.tensors[0].std()}")

    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])
