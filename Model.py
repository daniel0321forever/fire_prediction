from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS

from Dataset import HousePriceDataset_v2


"""
NOTE
"""

"""
Note 2
- Never use pandas.getdummies, but use sckitlearn.preprocessing instead
- Use Target Encoding instead
- Do note that there is nothing stopping you from adding a .device property to the models.
"""


class HousePriceDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32) -> None:
        super().__init__()
        self.batch_size = batch_size


    def setup(self, stage):
        # TODO Adjust dataset
        dataset = HousePriceDataset_v2(stage="train", modify="apply")
        pred_dataset = HousePriceDataset_v2(stage="test", modify="apply")

        self.train_set, self.val_set, self.test_set = random_split(dataset, [0.7,0.2,0.1])
        self.pred_set = pred_dataset
    
    def train_dataloader(self):
        train_set = DataLoader(self.train_set, batch_size=self.batch_size)
        return train_set
    
    def val_dataloader(self):
        val_set = DataLoader(self.val_set, batch_size=self.batch_size)
        return val_set
    
    def test_dataloader(self):
        test_set = DataLoader(self.test_set)
        return test_set
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pred_set = DataLoader(self.pred_set)
        return pred_set

    def input_dim(self):
        dataset = HousePriceDataset_v2()
        return dataset.dim()
    
class DNN(L.LightningModule):
    def __init__(self, **config) -> None:
        super().__init__()

        # TODO: take configruation from config file only
        config.setdefault("input_dim", 0)
        config.setdefault("output_dim", 1)
        config.setdefault("hiddens", [128, 64, 32])
        config.setdefault("drop_rate", 0.3)
        config.setdefault("slope", 0.1)

        hiddenLayersList = []
        for i in range(len(config["hiddens"]) - 1):
            hiddenLayersList.append(nn.Linear(config["hiddens"][i], config["hiddens"][i+1]))
            hiddenLayersList.append(nn.LeakyReLU(config["slope"]))
            hiddenLayersList.append(nn.Dropout(config["drop_rate"]))

        self.inputLayer = nn.Sequential(
            nn.Linear(config["input_dim"], config["hiddens"][0]),
            nn.LeakyReLU(negative_slope=config["drop_rate"])
        )
        self.hiddenLayers = nn.ModuleList(hiddenLayersList)
        self.outputLayer = nn.Sequential(
            nn.Linear(config["hiddens"][-1], config["output_dim"]),
        )

        self.train_output = {"loss": []}
        self.val_output = {"loss": [], 'best_loss': 20000}
        self.test_len = 0
        self.test_loss = 0

        self.config = config
        self.save_hyperparameters()

    def forward(self, x) -> Any:
        # It would be called when we run the class or explicilty call foward method. The training 
        # step would call this method to get the output of the model and find loss
        x = self.inputLayer(x)
        for layer in self.hiddenLayers:
            x = layer(x)
        
        x = nn.Sigmoid(x)
        y = self.outputLayer(x)
        return y
    
    def training_step(self, batch, batch_idx):
        # return the loss, lightning would do backward propogation 
        # when training depending on the loss from the training step
        
        # TODO: change input variable
        input, fire = batch
        fire = fire.unsqueeze(1)
        fire_pred = self(input)

        # TODO: Use Cross Entropy loss
        mse_loss = nn.CrossEntropyLoss
        loss = mse_loss(fire, fire_pred)
        self.log("train_loss", loss, on_epoch=True)

        self.train_output["loss"].append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # when it is defined, lightning module would call it in each step of training
        input, price = batch
        price = price.unsqueeze(1)
        price_hat = self(input)
        mse_loss = nn.MSELoss()
        loss = mse_loss(price, price_hat)
        if loss < self.val_output['best_loss']:
            self.val_output['best_loss'] = loss

        self.log("val_loss", loss, on_epoch=True)
        self.val_output["loss"].append(loss)
    
    def test_step(self, batch, batch_idx):
        # when it is defined, lightning module would call it when Trainer.test() is called
        input, price = batch
        price = price.unsqueeze(1)
        price_hat = self(input)
        mse_loss = nn.MSELoss()
        loss = mse_loss(price, price_hat).item()

        self.test_loss += loss
        self.test_len += 1

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        input, id = batch
        return self(input)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["init_lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=self.config["sched_factor"], 
            patience=self.config["sched_patience"], 
            verbose=True
        )
        return optimizer