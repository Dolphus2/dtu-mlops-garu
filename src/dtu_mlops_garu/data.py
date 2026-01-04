from pathlib import Path
import typer
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt  # only needed for plotting
from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

N_TRAIN_FILES = 5

RAW_DATA_PATH = Path("data") / "raw" / "corruptmnist" 
PROCESSED_DATA_PATH = Path("data") / "processed" / "corruptmnist" 

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

@data_app.command()
def corrupt_mnist(data_path: Path) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
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


    train_images: torch.tensor = torch.load(f"{data_path}/train_images.pt")
    train_target: torch.tensor = torch.load(f"{data_path}/train_target.pt")

    test_images: torch.Tensor = torch.load(f"{data_path}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{data_path}/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set

def get_dataloaders(data_path: Path, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    training_data, test_set = corrupt_mnist(data_path)
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
    dims = tuple(range(0, images.ndim)) # Wrong I guess. Should be over all images
    train_normalized = images - images.mean(dims, keepdim=True)
    train_normalized /= train_normalized.std(dims, keepdim=True)
    return images

@data_app.command()
def preprocess_mnist(src: Path, dst: Path) -> None:
    """Preprocess data from data_path to DST."""
    print("Preprocessing data...")
    train_images, train_target = [], []
    for i in range(N_TRAIN_FILES + 1):
        train_images.append(torch.load(f"{src}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{src}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{src}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{src}/test_target.pt")

    train_images = train_images.unsqueeze(1).float() ## add an extra dimension
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    dst.mkdir(exist_ok=True)
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
    typer.run(preprocess_mnist)
    data_path = Path("data/raw/corruptmnist")
    output_folder = Path("data/processed/corruptmnist")
    # preprocess(data_path, output_folder)
    train_set, test_set = corrupt_mnist(data_path)
    print(f"Size of training set: {len(train_set)}")
    print(f"Size of test set: {len(test_set)}")
    print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
    print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
    show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])
