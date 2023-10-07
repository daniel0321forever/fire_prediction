from typing import Any, Optional

import torch
from torch import nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import binary_accuracy

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS

import Dataset
from Dataset import FPDataset



"""
Note 2
- Never use pandas.getdummies, but use sckitlearn.preprocessing instead
- Do note that there is nothing stopping you from adding a '.device' property to the models.
"""

class FirePredcitDM(L.LightningDataModule):
    def __init__(self, batch_size=32, mode="DNN") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.mode = mode # DNN, RNN

    def setup(self, stage):
        dataset = FPDataset(stage="train", mode=self.mode)
        pred_dataset = FPDataset(stage="test", mode=self.mode)

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
        dataset = FPDataset(mode=self.mode)
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
        
        x = nn.Sigmoid()(x)
        y = self.outputLayer(x)
        return y
    
    def training_step(self, batch, batch_idx):
        # return the loss, lightning would do backward propogation 
        # when training depending on the loss from the training step
        
        # TODO: change input variable
        input, is_fire = batch
        is_fire = is_fire.unsqueeze(1)
        is_fire_pred = self(input)

        # TODO: Use Cross Entropy loss
        c_e_loss = nn.BCELoss()
        loss = c_e_loss(is_fire_pred, is_fire)
        self.log("train_loss", loss, on_epoch=True)

        self.train_output["loss"].append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # when it is defined, lightning module would call it in each step of training
        input, is_fire = batch
        is_fire = is_fire.unsqueeze(1)
        is_fire_pred = self(input)
        c_e_loss = nn.BCELoss()
        loss = c_e_loss(is_fire_pred, is_fire)
        if loss < self.val_output['best_loss']:
            self.val_output['best_loss'] = loss

        self.log("val_loss", loss, on_epoch=True)
        self.val_output["loss"].append(loss)
    
    def test_step(self, batch, batch_idx):
        # when it is defined, lightning module would call it when Trainer.test() is called
        input, is_fire = batch
        is_fire = is_fire.unsqueeze(1)
        is_fire_pred = self(input)
        c_e_loss = nn.BCELoss()
        loss = c_e_loss(is_fire_pred, is_fire).item()

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
    
class FirPRNN(L.LightningModule):
    def __init__(self, **config) -> None:
        super().__init__()

        config.setdefault("input_dim_hours", 0)
        config.setdefault("input_dim_days", 1)
        config.setdefault("output_dim", 1)
        config.setdefault("rnn_hidden", 32)
        config.setdefault("hiddens", [128])
        config.setdefault("drop_rate", 0.3)
        config.setdefault("slope", 0.1)

        # RNN layers
        self.hours_lstm = torch.nn.LSTM(
            config["input_dim_hours"], # the input dim of each time data
            config["rnn_hidden"], 
            config["rnn_layers"], 
            proj_size = config["rnn_proj"] # output dim of each time data
        )

        self.days_lstm = torch.nn.LSTM(
            config["input_dim_days"], # the input dim of each time data
            config["rnn_hidden"], 
            config["rnn_layers"], 
            proj_size = config["rnn_proj"] # output dim of each time data
        )
        
        # dense layers
        denseLayerList = []
        self.inputLayer = nn.Sequential(
            nn.Linear(config["rnn_proj"], config["hiddens"][0]),
            nn.LeakyReLU(config["slope"]),
            nn.Dropout(config["drop_rate"]),
        )
        for i in range(len(config["hiddens"]) - 1):
            denseLayerList.append(nn.Linear(config["hiddens"][i], config["hiddens"][i+1]))
            denseLayerList.append(nn.LeakyReLU(config["slope"]))
            denseLayerList.append(nn.Dropout(config["drop_rate"]))
        self.DenseLayers = nn.ModuleList(denseLayerList)

        self.outputLayer = nn.Sequential(
            nn.Linear(config["hiddens"][-1], config["output_dim"]),
            nn.Sigmoid(),
        )
        
        # output metrics
        self.train_output = {"loss": []}
        self.val_output = {"loss": [], "acc": [], 'best_loss': 20000}
        self.test_len = 0
        self.test_loss = 0
        
        # save config
        self.config = config
        self.save_hyperparameters()

    def forward(self, x_h) -> Any:
        # It would be called when we run the class or explicilty call foward method. The training 
        # step would call this method to get the output of the model and find loss
        # TODO: Take two parts of input respectively, and get output out of it
        x_h, (h, c) = self.hours_lstm(x_h)
        # print("x_h shape", x_h.shape)
        x_final_h = x_h.select(dim=1, index=-1) # get the data from the final hours
        # print("x_final_h shape", x_final_h.shape)

        # TODO: concat
        x_concat = x_final_h
        # pass

        x = self.inputLayer(x_concat)
        for layer in self.DenseLayers:
            x = layer(x)
        y = self.outputLayer(x)

        return y
    
    def training_step(self, batch, batch_idx):
        # return the loss, lightning would do backward propogation 
        # when training depending on the loss from the training step
        
        # TODO: later we should add daily input
        input_hours, is_fire = batch
        is_fire = is_fire.unsqueeze(1)
        is_fire_pred = self(input_hours)

        # loss
        c_e_loss = nn.BCELoss()
        loss = c_e_loss(is_fire_pred, is_fire)

        # acc
        acc = binary_accuracy(is_fire_pred[-1], is_fire[-1], threshold=0.5)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        self.train_output["loss"].append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # when it is defined, lightning module would call it in each step of training
        input_hours, is_fire = batch
        is_fire = is_fire.unsqueeze(1)
        is_fire_pred = self(input_hours)

        # loss
        c_e_loss = nn.BCELoss()
        loss = c_e_loss(is_fire_pred, is_fire)
        if loss < self.val_output['best_loss']:
            self.val_output['best_loss'] = loss

        acc = binary_accuracy(is_fire_pred.squeeze(1), is_fire.squeeze(1), threshold=0.5)

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

        self.val_output["loss"].append(loss)
        self.val_output["acc"].append(acc)

    def test_step(self, batch, batch_idx):
        # when it is defined, lightning module would call it when Trainer.test() is called
        input_hours, is_fire = batch
        is_fire = is_fire.unsqueeze(1)
        is_fire_pred = self(input_hours)
        
        # loss
        c_e_loss = nn.BCELoss()
        loss = c_e_loss(is_fire_pred, is_fire).item()

        # logging
        self.test_loss += loss
        self.test_len += 1

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        input = batch
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