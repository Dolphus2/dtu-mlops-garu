import pytest
import torch
from torch.utils.data import Dataset

from dtu_mlops_garu.data import PROCESSED_DATA_PATH, MyDataset, get_corrupt_mnist, get_dataloaders

N_TRAIN = 30000
N_TEST = 5000
k = 42
BATCH_SIZE = 64
DATA_PRESENT = not PROCESSED_DATA_PATH.exists() or not (PROCESSED_DATA_PATH / "train_images.pt").is_file()


def np_close(x: float, y: float, eps: float = 1e-5):
    return torch.abs(x - y) < eps


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)


@pytest.mark.skipif(DATA_PRESENT, reason="Data files not found")
def test_data():
    train_set, test_set = get_corrupt_mnist(PROCESSED_DATA_PATH)
    assert len(train_set) == N_TRAIN, "Dataset did not have the correct number of samples"
    assert len(test_set) == N_TEST, "Dataset did not have the correct number of samples"
    for dataset in [train_set, test_set]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)

    train_targets = torch.unique(train_set.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all()
    test_targets = torch.unique(test_set.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all()

    assert np_close(train_set.tensors[0].mean(), 0)
    assert np_close(train_set.tensors[0].std(), 1)

    assert np_close(test_set.tensors[0].mean(), 0)
    assert np_close(test_set.tensors[0].std(), 1)


def random_samples(data_set):
    N = len(data_set)
    assert k <= N
    samples = torch.randperm(N)[:k]
    for X, y in zip(*data_set[samples]):
        _test_shape(X, y)


def _test_shape(X, y):
    assert X.shape == torch.Size([1, 28, 28])
    assert y.shape == torch.Size([])


def test_dataloader():
    train_dataloader, test_dataloader = get_dataloaders(PROCESSED_DATA_PATH, BATCH_SIZE)
    for batch, (X, y) in enumerate(train_dataloader):
        assert X.shape == torch.Size([BATCH_SIZE, 1, 28, 28])
        assert y.shape == torch.Size([BATCH_SIZE])
        if batch >= 10:
            break

    for batch, (X, y) in enumerate(test_dataloader):
        assert X.shape == torch.Size([BATCH_SIZE, 1, 28, 28])
        assert y.shape == torch.Size([BATCH_SIZE])
        if batch >= 10:
            break


if __name__ == "__main__":
    test_data()
