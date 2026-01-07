import torch
from pytorch_lightning import LightningModule
from torch import nn, optim


class Model2(LightningModule):
    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.lr = 1e-3
        self.criterion = nn.CrossEntropyLoss()

        self.p = dropout
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(self.p)
        self.fc1 = nn.Linear(128, 10)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        return torch.max_pool2d(x, 2, 2)

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.backbone(x)
        return self.classifier(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(x)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        logits = self(inputs)
        loss = self.criterion(logits, target)
        acc = (logits.argmax(dim=1) == target).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class Model1(nn.Module):
    def __init__(self, c1=128, out_features=10, dropout=0.5):
        super().__init__()
        self.c1 = c1
        self.dropout = dropout
        self.out_features = out_features

        self.features = nn.Sequential(
            nn.Conv2d(1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.c1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.c1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c1, (3, 3)),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32768, self.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.features(x)


if __name__ == "__main__":
    model = Model1()
    print(f"Model architecture: {model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    x = torch.rand(64, 1, 28, 28)
    print(f"Output shape of model: {model(x).shape}")
