from typing import Any

import lightning as L
import numpy as np
import torch

from agatston.models.architectures.RaUNet import AttentionResidualUNet
from lightning.pytorch.cli import OptimizerCallable
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from torch.optim.lr_scheduler import OneCycleLR


class RaUNet3D(L.LightningModule):
    def __init__(self, optimizer: OptimizerCallable = torch.optim.AdamW):
        super(RaUNet3D, self).__init__()
        self.optimizer = optimizer
        self._model = AttentionResidualUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

        self.save_hyperparameters(self.hparams, ignore=['criterion'])
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True, weight=torch.tensor([1, 3, 1, 1]))
        self.metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)
        self.optimizer = optimizer
        self.post_pred = Compose([EnsureType('tensor'),
                                    AsDiscrete(argmax=True, to_onehot=4, num_classes=4)])
        self.post_label = Compose([EnsureType('tensor'),
                                     AsDiscrete(to_onehot=4, num_classes=4)])
        self.best_val_dice = 0.0
        self.best_val_epoch = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self) -> Any:
        optimizer = self.optimizer(self.parameters())
        steps_per_epoch = 45000 // 256
        scheduler_dict = {
            'scheduler': OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            'interval': 'step',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

    def training_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']

        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        metric = self.metric(outputs, labels)
        logs = {'train_loss': loss.item(), 'train_dice': metric}
        d = {'loss': loss, 'log': logs, 'progress_bar': logs}
        self.training_step_outputs.append(d)
        return d

    def on_train_epoch_end(self) -> None:
        train_loss = np.mean([x['log']['train_loss'] for x in self.training_step_outputs])
        train_dice = np.mean([torch.Tensor.cpu(x['log']['train_dice']) for x in self.training_step_outputs])
        self.log('train_loss', train_loss, sync_dist=True)
        self.log('train_dice', train_dice, sync_dist=True)
        self.logger.log_metrics({'train_loss': train_loss, 'train_dice': train_dice}, step=self.trainer.current_epoch)
        self.training_step_outputs.clear()

    def evaluation_step(self, batch):
        images, labels = batch['images'], batch['labels']
        roi_size = (96, 96, 96)
        sw_batch_size = 1
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self._model)
        loss = self.loss_function(outputs, labels)
        metric = self.metric(y_pred=outputs, y=labels)
        return outputs, loss, metric, len(outputs)

    def validation_step(self, batch, batch_idx):
        outputs, loss, metric, num_items = self.evaluation_step(batch)
        self.validation_step_outputs.append({"val_loss": loss, "val_number": num_items})
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
        return {'log': logs}

    def on_validation_epoch_end(self) -> None:
        self.evaluation_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()
