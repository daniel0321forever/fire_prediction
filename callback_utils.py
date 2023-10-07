import lightning.pytorch as pl
import torch

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
from lightning.pytorch.callbacks import EarlyStopping, Callback, ModelCheckpoint


class PrintingCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        loss = torch.stack(pl_module.train_output['loss']).mean()
        # print(f"\n\ntraining loss: {loss:.3f}")
        pl_module.train_output['loss'].clear()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        loss = torch.stack(pl_module.val_output['loss']).mean()
        acc = torch.stack(pl_module.val_output['acc']).mean()

        best = pl_module.val_output['best_loss']
        print(f"\n\n\nval loss {loss:.3f}, val acc {acc:.3f}, best val loss {best:.3f}")
        pl_module.val_output['loss'].clear()
        pl_module.val_output['acc'].clear()

class TestingCallback(Callback):
    def on_test_end(self, trainer, pl_module) -> None:
        print(f"The avg loss is {pl_module.test_loss / pl_module.test_len}")