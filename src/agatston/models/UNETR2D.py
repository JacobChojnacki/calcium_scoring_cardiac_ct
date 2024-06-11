import torch

from agatston.models.architectures.UNETR import UNETR
from agatston.models.RaUNet2D import RaUNet2D
from lightning.pytorch.cli import OptimizerCallable


class UNETR2D(RaUNet2D):
    def __init__(self, optimizer: OptimizerCallable = torch.optim.AdamW):
        super().__init__(optimizer)
        self.optimizer = optimizer
        self._model = UNETR(
            spatial_dims=2,
            in_channels=1,
            img_size=512,
            out_channels=4,
            feature_size=8,
            hidden_size=384,
            mlp_dim = 1536,
            num_heads=6
        )

        self.save_hyperparameters(ignore=['criterion'])
