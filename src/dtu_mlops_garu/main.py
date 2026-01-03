from pathlib import Path

import torch
import typer
from dtu_mlops_garu.data import data_app
from dtu_mlops_garu.train import train_app
from dtu_mlops_garu.evaluate import evaluate_app
from dtu_mlops_garu.visualize import visualize_app

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
RAW_DATA_PATH = Path("data") / "raw" / "corruptmnist"
PROCESSED_DATA_PATH = Path("data") / "processed" / "corruptmnist"

app = typer.Typer()
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(visualize_app, name="visualize")

if __name__ == "__main__":
    app()
