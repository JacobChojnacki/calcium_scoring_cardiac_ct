import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    SqueezeDimd,
    ToTensord,
)


class Transformations3D:
    def __init__(self):
        self.transform = Compose(
            [
                LoadImaged(keys=['images', 'labels']),
                EnsureTyped(keys=['images', 'labels']),
                EnsureChannelFirstd(keys=['images', 'labels']),
                RandCropByPosNegLabeld(
                    keys=['images', 'labels'],
                    label_key='labels',
                    spatial_size=(96, 96, 96),
                    pos=4,
                    neg=1,
                    num_samples=4,
                    image_key='images',
                    image_threshold=0,
                ),
                RandShiftIntensityd(keys='images', offsets=0.02, prob=0.25),
                RandAdjustContrastd(keys='images', gamma=(0.98, 1.02), prob=0.25),
                ToTensord(keys=['images', 'labels']),
            ]
        )

    def __call__(self, data):
        return self.transform(data)


class TransformationsVal:
    def __init__(self):
        self.transform = Compose(
            [
                LoadImaged(keys=['images', 'labels']),
                EnsureTyped(keys=['images', 'labels'], dtype=torch.float),
                EnsureChannelFirstd(keys=['images', 'labels']),
                ToTensord(keys=['images', 'labels']),
            ]
        )

    def __call__(self, data):
        return self.transform(data)


class Transformations2D:
    def __init__(self):
        self.transform = Compose(
            [
                LoadImaged(keys=['images', 'labels', 'raw_plaques']),
                EnsureTyped(keys=['images', 'labels']),
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
                ToTensord(keys=['images', 'labels']),
            ]
        )

    def __call__(self, data):
        return self.transform(data)
