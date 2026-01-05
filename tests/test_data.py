import torch
from torch.utils.data import Dataset
from dtu_mlops_garu.data import *
N_TRAIN = 30000
N_TEST = 5000
k = 42
BATCH_SIZE = 64

def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)


def test_data():
    train_set, test_set = get_corrupt_mnist(PROCESSED_DATA_PATH)
    assert len(train_set) == N_TRAIN
    random_samples(train_set)
    _test_labels(train_set)
    
    assert len(test_set) == N_TEST
    random_samples(test_set)
    _test_labels(test_set)

def random_samples(data_set):
    N = len(data_set)
    assert N >= k
    samples = torch.randperm(N)[:k]
    for X, y in zip(*data_set[samples]):
        _test_shape(X, y)

def _test_shape(X, y):
    assert X.shape == torch.Size([1,28,28])
    assert y.shape == torch.Size([])


def _test_labels(data_set):
    y = data_set.tensors[1]
    assert max(y) == 9
    assert len(torch.unique(y)) == 10
    assert (torch.unique(y) == torch.arange(0,10)).all()

def test_dataloader():
    train_dataloader, test_dataloader = get_dataloaders(PROCESSED_DATA_PATH, BATCH_SIZE)
    for batch, (X, y) in enumerate(train_dataloader):
        assert X.shape == torch.Size([BATCH_SIZE,1,28,28])
        assert y.shape == torch.Size([BATCH_SIZE])
        if batch >= 10: break

    for batch, (X, y) in enumerate(test_dataloader):
        assert X.shape == torch.Size([BATCH_SIZE,1,28,28])
        assert y.shape == torch.Size([BATCH_SIZE])
        if batch >= 10: break

if __name__ == "__main__":
    test_data()