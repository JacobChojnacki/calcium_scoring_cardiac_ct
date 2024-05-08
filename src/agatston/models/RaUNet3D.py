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
from monai.transforms import AsDiscrete, Compose, EnsureType
from torch.optim.lr_scheduler import OneCycleLR


class RaUNet(L.LightningModule):
    def __init__(self, optimizer: OptimizerCallable = torch.optim.AdamW):
        super(RaUNet, self).__init__()
        self.optimizer = optimizer
        self._model = AttentionResidualUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

        self.save_hyperparameters(self.hparams, ignore=['criterion'])
        self.__loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.__metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)
        self.optimizer = optimizer
        self.__post_pred = Compose([EnsureType('tensor', device='cpu'),
                                    AsDiscrete(argmax=True, to_onehot=4, num_classes=4)])
        self.__post_label = Compose([EnsureType('tensor', device='cpu'),
                                     AsDiscrete(to_onehot=4, num_classes=4)])
        self.__best_val_dice = 0.0
        self.__best_val_epoch = 0
        self.__training_step_outputs = []
        self.__validation_step_outputs = []
        self.__test_step_outputs = []

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
        loss = self.__loss_function(outputs, labels)
        metric = self.__metric(outputs, labels)
        logs = {'train_loss': loss.item(), 'train_dice': metric}
        d = {'loss': loss, 'log': logs, 'progress_bar': logs}
        self.__training_step_outputs.append(d)
        return d

    def on_train_epoch_end(self) -> None:
        train_loss = np.mean([x['log']['train_loss'].item() for x in self.__training_step_outputs])
        train_dice = np.mean([x['log']['train_dice'] for x in self.__training_step_outputs])
        self.log('train_loss', train_loss, sync_dist=True)
        self.log('train_dice', train_dice, sync_dist=True)
        self.logger.log_metrics({'train_loss': train_loss, 'train_dice': train_dice}, step=self.trainer.current_epoch)
        self.__training_step_outputs.clear()

    def evaluation_step(self, batch):
        images, labels = batch['images'], batch['labels']
        labels = labels.to(torch.float32)
        roi_size = (96, 96, 96)
        sw_batch_size = 1
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self._model)
        outputs = outputs.to(torch.float32)
        loss = self.__loss_function(outputs, labels)
        outputs = [self.__post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.__post_label(i) for i in decollate_batch(labels)]
        metric = self.__metric(outputs, labels)
        return {'val_loss': loss, 'val_dice': metric, 'val_number': len(outputs)}

    def validation_step(self, batch, batch_idx):
        outputs = self.evaluation_step(batch)
        self.__validation_step_outputs.append(outputs)
        return outputs

    def evaluation_epoch_end(self, outputs, split='val'):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output[f'{split}_loss'].sum().item()
            num_items += output[f'{split}_number']

        mean_val_metric = self.__metric.aggregate().item()
        self.__metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        logs = {
            f'{split}_loss': mean_val_loss,
            f'{split}_dice': mean_val_metric,
        }
        if mean_val_metric > self.__best_val_dice:
            self.__best_val_dice = mean_val_metric
            self.__best_val_epoch = self.current_epoch
        print(
            f'Current epoch: {self.current_epoch}, Best epoch: {self.__best_val_epoch}, \
                Best val dice: {self.__best_val_dice} at epoch {self.__best_val_epoch}'
        )
        self.log_dict(logs, sync_dist=True)
        self.logger.log_metrics(logs, step=self.trainer.current_epoch)
        return {'log': logs, 'progress_bar': logs}

    def on_validation_epoch_end(self) -> None:
        self.evaluation_epoch_end(self.__validation_step_outputs, 'val')
        self.__validation_step_outputs.clear()
