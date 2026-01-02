from torch import nn
import torch

class Model(nn.Module):
    def __init__(self, c1 = 128, out_features = 10):
        super().__init__()
        self.c1 = c1
        self.p = 0.5
        self.out_features = out_features

        self.features = nn.Sequential(
            nn.Conv2d(1, self.c1, (3,3)),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Conv2d(self.c1, self.c1, (3,3)),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c1, (3,3)),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c1, (3,3)),
            nn.ReLU(),
            nn.Dropout(self.p),
            nn.Conv2d(self.c1, self.c1, (3,3)),
            nn.ReLU(),
            nn.Conv2d(self.c1, self.c1, (3,3)),
            nn.ReLU(),
            )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768, self.out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = Model()
    x = torch.rand(64, 1, 28, 28)
    print(f"Output shape of model: {model(x).shape}")
