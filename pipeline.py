import Dataset
import Model
from callback_utils import PrintingCallback, TestingCallback

import lightning as L

import torch
from torchsummary import summary
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import lightning as L
import pandas as pd
import yaml


def train(epoch=10, data_module=Dataset.FPDataset(), config_dir="config.yaml"):
    # dataset
    input_dim = data_module.input_dim()

    # configuration
    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["input_dim"] = input_dim

    print("==================Training Configuration===================")
    for x, y in config.items():
        print(f"{x}: {y}")
    print("===========================================================")

    # build model
    model = Model.DNN(**config)
    summary(model)

    # callback
    MyEarlyStopping = EarlyStopping(
        monitor="val_loss",  # monitor string is the label in logging
        min_delta=0,
        patience=config["earlystopping_patience"],
        mode='min'
    )

    SaveBestCheckpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename='BESTcheckpoint--{epoch}-{val_loss:.2f}',
        save_top_k=1,
    )

    # train
    trainer = L.Trainer(
        max_epochs=epoch, 
        default_root_dir=OUTPUT_DIR, 
        accelerator=ACCELERATOR, 
        callbacks=[PrintingCallback(), MyEarlyStopping, SaveBestCheckpoint],
        devices="auto")
    trainer.fit(model, data_module) # the data from module would be move to the same device as model defaulty by lightning
