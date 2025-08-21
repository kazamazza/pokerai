# ml/trainers/train_populationnet.py
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from ml.data.population_dataset import PopulationDatasetParquet
from ml.models.populationnet import PopulationNetLit


def run_train(
    parquet_path: str = "data/datasets/populationnet_nl10.parquet",
    batch_size: int = 2048,
    max_epochs: int = 5,
    lr: float = 1e-3,
    num_workers: int = 0,
):
    ds = PopulationDatasetParquet(parquet_path)

    # derive cards from dataset (it should expose this)
    cards = ds.categorical_cardinalities()  # {"stakes_id": 4, "street_id":4, ...}
    feature_order = ds.x_cols                # keep same order

    # stratified split (your dataset exposes this)
    train_idx, val_idx = ds.stratified_indices(train_frac=0.8, seed=42)
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = PopulationNetLit(cards=cards, feature_order=feature_order, lr=lr)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )
    trainer.fit(model, train_dl, val_dl)

if __name__ == "__main__":
    run_train()