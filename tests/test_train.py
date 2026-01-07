import torch

import wandb
from dtu_mlops_garu.data import PROCESSED_DATA_PATH, get_dataloaders
from dtu_mlops_garu.model import Model2
from dtu_mlops_garu.utils.train_utils import train_epoch

BATCH_SIZE = 64
LR = 1e-3
MOMENTUM = 0.9
wandb.init(mode="disabled")


def test_train():
    model = Model2()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    train_dataloader, test_dataloader = get_dataloaders(PROCESSED_DATA_PATH, BATCH_SIZE)

    for dataloader in [train_dataloader, test_dataloader]:
        epoch_stats = train_epoch(dataloader, model, criterion, optimizer, device="cpu")
