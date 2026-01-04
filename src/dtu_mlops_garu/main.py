from pathlib import Path

import typer
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

from dtu_mlops_garu.model import Model2
from dtu_mlops_garu.data import data_app, get_dataloaders
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

@hydra.main(version_base=None, config_path="../../config", config_name="train_config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    batch_size = cfg.get("batch_size", 64)
    

    train_dataloader, test_dataloader = get_dataloaders(PROCESSED_DATA_PATH, batch_size=batch_size)
    model = Model2(cfg.model.get("dropout", 0.5))
    early_stopping = EarlyStopping(
        monitor="train_loss", patience=3, verbose=True, mode="min"
    )

    checkpoint = ModelCheckpoint(
    dirpath=Path.cwd() / "models",
    monitor="val_acc",
    mode="max",
    save_top_k=3,  # Save 3 best models
    filename="model-{epoch:02d}-{val_acc:.2f}",
)

    trainer = Trainer(
        accelerator="gpu", 
        default_root_dir=Path.cwd() / "models", 
        callbacks=[early_stopping, checkpoint],
        max_epochs=20,
        limit_train_batches=0.2,
        logger=WandbLogger(project="dtu_mlops_garu"),
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader,)

if __name__ == "__main__":
    main()
