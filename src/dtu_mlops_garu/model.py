import torch
from torch import nn


class Model1(nn.Module):
    def __init__(self, c1=128, out_features=10):
        super().__init__()
        self.c1 = c1
        self.p = 0.5
        self.out_features = out_features

        self.features = nn.Sequential(
            nn.Conv2d(1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Conv2d(self.c1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c1, (3, 3)),
            nn.ReLU(),
            nn.Dropout(self.p),
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


class Model2(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = Model1()
    print(f"Model architecture: {model}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    x = torch.rand(64, 1, 28, 28)
    print(f"Output shape of model: {model(x).shape}")
