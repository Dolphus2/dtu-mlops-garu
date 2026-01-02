import numpy as np
import torch

def train_epoch(dataloader, model, criterion, optimizer, device, print_freq = 100):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    statistics = {"train_loss": [], "train_accuracy": []}
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        statistics["train_loss"].append(loss.item())

        accuracy = (pred.argmax(1) == y).type(torch.float).mean().item()
        statistics["train_accuracy"].append(accuracy)

        if batch % print_freq == 0:
            loss, current = loss.item(), (batch + 1) * len(X) # Because you start at zero and batches are counted in batches, not items, so multiply by X. 
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return statistics

def test(dataloader, model, criterion, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
