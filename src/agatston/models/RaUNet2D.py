import torch

from agatston.models.architectures.RaUNet import AttentionResidualUNet
from agatston.models.RaUNet3D import RaUNet3D
from lightning.pytorch.cli import OptimizerCallable
from monai.inferers import SliceInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


class RaUNet2D(RaUNet3D):
    def __init__(self, optimizer: OptimizerCallable = torch.optim.AdamW):
        super().__init__()
        self.optimizer = optimizer
        self._model = AttentionResidualUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )

        self.save_hyperparameters(self.hparams, ignore=['criterion'])
        self.save_hyperparameters(self.hparams, ignore=['criterion'])
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)
        self.optimizer = optimizer
        self.best_val_dice = 0.0
        self.best_val_epoch = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        images, labels = batch['images'], batch['labels']
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        logs = {"train_loss": loss.item(), "train_dice": self.metric(output, labels)}
        d = {"loss": loss, "log": logs}
        self.training_step_outputs.append(d)
        return d

    def evaluation_step(self, batch):
        images, labels = batch['images'], batch['labels']
        roi_size = (512, 512)
        sw_batch_size = 1
        slice_inferer = SliceInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            spatial_dim=2,
            progress=True,
            padding_mode='replicate'
        )
        outputs = slice_inferer(images, self._model)
        loss = self.loss_function(outputs, labels)
        metric = self.metric(y_pred=outputs > 0.5, y=labels)
        return outputs, loss, metric, len(outputs)
