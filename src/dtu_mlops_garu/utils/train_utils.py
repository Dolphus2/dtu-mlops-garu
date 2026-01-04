import torch
from dtu_mlops_garu.utils.wandb_utils import log_images, log_ROC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_epoch(dataloader, model, criterion, optimizer, device, print_freq=100):
    size = len(dataloader.dataset)
    model.train()

    preds, ys = [], []
    epoch_stats = {"train_loss": [], "train_accuracy": []}
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        epoch_stats["train_loss"].append(loss.item())

        accuracy = (pred.argmax(1) == y).type(torch.float).mean().item()
        epoch_stats["train_accuracy"].append(accuracy)

        preds.append(pred.detach().cpu())
        ys.append(y.detach().cpu())
        if batch % print_freq == 0:
            loss, current = (
                loss.item(),
                (batch + 1) * len(X),
            )  # Because you start at zero and batches are counted in batches, not items, so multiply by X.
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            log_images(model, X)
        

    # Concatenate all predictions and labels and log ROC plot
    preds = torch.cat(preds, dim=0)
    ys = torch.cat(ys, dim=0)

    log_ROC(ys, preds)
    
    # Compute metrics
    precision = precision_score(ys, preds.argmax(dim=1), average="weighted")
    recall = recall_score(ys, preds.argmax(dim=1), average="weighted")
    f1 = f1_score(ys, preds.argmax(dim=1), average="weighted")
    
    # Add metrics to epoch_stats
    epoch_stats["precision"] = precision
    epoch_stats["recall"] = recall
    epoch_stats["f1"] = f1

    return epoch_stats


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
