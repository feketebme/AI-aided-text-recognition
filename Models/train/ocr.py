from Models.architectures.ocr_arc import TextDetectLSTM
from Models.base_model import PlBaseModel
from Data.dataloader_base import DataProvider
from Data.Datasets.dataset_ocr import OCRDataset
from Data.utils_data import CTCLoss,CERMetrics
from utils import initialize_with_config
import torch.nn as nn

from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint, EarlyStopping
import argparse
import pytorch_lightning as pl


def get_loss_fn():
    _ = CTCLoss(blank=0)
    return _


def get_metrics(vocab):
    _ = CERMetrics(vocab)
    return _


def get_callbacks(monitor):
    _ = [
        ModelSummary(),
        ModelCheckpoint(monitor=monitor, dirpath="Models/train/weights/OCR"),
        EarlyStopping(monitor=monitor, patience=5),
    ]
    return _


def setup_and_run(config, max_epochs):
    data_loader = initialize_with_config(DataProvider, config_file=config, dataset_object=OCRDataset, type="data")
    network = initialize_with_config(TextDetectLSTM, config_file=config, type="model", input_dim=3,
                                     output_dim=len(data_loader.available_datasets[0].vocab))
    model = initialize_with_config(PlBaseModel, config_file=config,type="model", backbone=network, loss_fn=get_loss_fn(),
                                   metrics=get_metrics(data_loader.available_datasets[0].vocab))
    trainer = pl.Trainer(
        max_epochs=int(max_epochs),
        devices="auto",
        accelerator="auto",
        callbacks=get_callbacks(model.monitor),
        logger=pl.loggers.TensorBoardLogger("logs/"),
    )
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize with config file")
    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    parser.add_argument("--max_epochs", type=int, help="", required=True)
    args = parser.parse_args()
    setup_and_run(config=args.config, max_epochs=args.max_epochs)
