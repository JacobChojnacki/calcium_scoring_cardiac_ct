import gc

from typing import Any

import lightning as L
import numpy as np
import torch

from agatston.models.architectures.RaUNet import AttentionResidualUNet
from lightning.pytorch.cli import OptimizerCallable
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
)


class RaUNet3D(L.LightningModule):
    def __init__(self, optimizer: OptimizerCallable = torch.optim.AdamW):
        super().__init__()
        self.optimizer = optimizer
        self._model = AttentionResidualUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

        self.save_hyperparameters(ignore=['criterion'])
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=4)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=4)])
        self.optimizer = optimizer
        self.best_val_dice = 0.0
        self.best_val_epoch = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        logs = {'train_loss': loss.item()}
        d = {'loss': loss, 'log': logs, 'progress_bar': logs}
        self.training_step_outputs.append(d)
        del images, labels, outputs, logs, d
        torch.cuda.empty_cache()
        gc.collect()

        return loss

    def on_train_epoch_end(self) -> None:
        train_loss = np.mean([x['log']['train_loss'] for x in self.training_step_outputs])
        self.log('train_loss', train_loss, sync_dist=True)
        self.logger.log_metrics({'train_loss': train_loss}, step=self.trainer.current_epoch)
        self.training_step_outputs.clear()
        # Free memory after epoch
        torch.cuda.empty_cache()
        gc.collect()

    def evaluation_step(self, images, labels, roi_size, sw_batch_size):
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self._model)
        loss = self.loss_function(outputs, labels)
        outputs_check = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels_check = [self.post_label(i) for i in decollate_batch(labels)]
        metric = self.metric(y_pred=outputs_check, y=labels_check)

        del images, labels, outputs_check, labels_check
        torch.cuda.empty_cache()
        gc.collect()
        return outputs, loss, metric, len(outputs)

    def validation_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']

        # Limit patch size for inference
        roi_size = (96, 96, 96)  # Adjust as needed
        sw_batch_size = 1
        outputs, loss, _, num_items = self.evaluation_step(images, labels, roi_size, sw_batch_size)

        self.validation_step_outputs.append({"val_loss": loss, "val_number": num_items})

        # Free memory for this batch
        del images, labels, loss
        torch.cuda.empty_cache()
        gc.collect()
        return outputs

    def evaluation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output['val_loss'].sum().item()
            num_items += output['val_number']

        mean_val_metric = self.metric.aggregate().item()
        self.metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        logs = {
            'val_loss': mean_val_loss,
            'val_dice': mean_val_metric,
        }
        if mean_val_metric > self.best_val_dice:
            self.best_val_dice = mean_val_metric
            self.best_val_epoch = self.current_epoch
        print(
            f'Current epoch: {self.current_epoch},\n Best epoch: {self.best_val_epoch}, \n\
                Best val dice: {self.best_val_dice} at epoch {self.best_val_epoch}'
        )
        self.log_dict(logs, sync_dist=True)
        self.logger.log_metrics(logs, step=self.trainer.current_epoch)
        # Free memory after epoch
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
        return {'log': logs}

    def on_validation_epoch_end(self) -> None:
        self.evaluation_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()

