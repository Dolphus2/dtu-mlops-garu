from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataloader import DataLoader
import typer

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
RAW_DATA_PATH = Path("data") / "raw" / "corruptmnist" 
PROCESSED_DATA_PATH = Path("data") / "processed" / "corruptmnist" 

app = typer.Typer()


if __name__ == "__main__":
    app()