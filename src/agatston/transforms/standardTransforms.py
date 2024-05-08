import numpy as np
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    SqueezeDimd,
    ToTensord,
)
from monai.utils import GridSamplePadMode


class Transformations3D:
    def __init__(self):
        self.transform = Compose(
            [
                LoadImaged(keys=['images', 'labels']),
                EnsureTyped(keys=['images', 'labels'], dtype=torch.float),
                EnsureChannelFirstd(keys=['images', 'labels']),
                RandCropByPosNegLabeld(
                    keys=['images', 'labels'],
                    label_key='labels',
                    spatial_size=(96, 96, 96),
                    pos=4,
                    neg=1,
                    num_samples=8,
                    image_key='images',
                    image_threshold=0,
                ),
                RandShiftIntensityd(keys='images', offsets=0.02, prob=0.25),
                RandAdjustContrastd(keys='images', gamma=(0.98, 1.02), prob=0.25),
                RandAffined(
                    keys=['images', 'labels'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0,
                    spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1),
                    padding_mode=GridSamplePadMode.REFLECTION,
                ),
                ToTensord(keys=['images', 'labels']),
            ]
        )

    def __call__(self, data):
        return self.transform(data)


TRANSFORMATIONS3D = Compose(
    [
        LoadImaged(keys=['images', 'labels']),
        EnsureChannelFirstd(keys=['images', 'labels']),
        RandCropByPosNegLabeld(
            keys=['images', 'labels'],
            label_key='labels',
            spatial_size=(96, 96, 96),
            pos=4,
            neg=1,
            num_samples=8,
            image_key='images',
            image_threshold=0,
        ),
        RandShiftIntensityd(keys='images', offsets=0.02, prob=0.25),
        RandAdjustContrastd(keys='images', gamma=(0.98, 1.02), prob=0.25),
        RandAffined(
            keys=['images', 'labels'],
            mode=('bilinear', 'nearest'),
            prob=1.0,
            spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi / 15),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode=GridSamplePadMode.REFLECTION,
        ),
        EnsureTyped(keys=['images', 'labels']),
        ToTensord(keys=['images', 'labels']),
    ]
)

TRANSFORMATIONS2D = Compose(
    [
        LoadImaged(keys=['images', 'labels', 'raw_plaques']),
        EnsureChannelFirstd(keys=['images', 'labels', 'raw_plaques']),
        RandShiftIntensityd(keys='images', offsets=0.02, prob=0.25),
        RandAdjustContrastd(keys='images', gamma=(0.98, 1.02), prob=0.25),
        RandCropByPosNegLabeld(
            keys=['images', 'labels', 'raw_plaques'],
            label_key='raw_plaques',
            spatial_size=[-1, -1, 1],
            pos=1,
            neg=0.25,
            num_samples=4,
            image_key='images',
        ),
        SqueezeDimd(keys=['images', 'labels'], dim=3),
        EnsureTyped(keys=['images', 'labels']),
        ToTensord(keys=['images', 'labels']),
    ]
)
