import torch

from agatston.models.architectures.UNet import UNet
from agatston.models.RaUNet2D import RaUNet2D
from lightning.pytorch.cli import OptimizerCallable
from monai.networks.layers import Norm


class UNet2D(RaUNet2D):
    def __init__(self, optimizer: OptimizerCallable = torch.optim.AdamW):
        super().__init__(optimizer)
        self.optimizer = optimizer
        self._model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )

        self.save_hyperparameters(ignore=['criterion'])
