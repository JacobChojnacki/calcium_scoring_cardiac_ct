from pathlib import Path
from typing import Callable, Optional

import lightning as L
import yaml

from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from monai.data import CacheDataset


def make_calcium_dataset(
    yaml_file_path: Path,
    is_training_set: bool = True,
    transforms: Callable = None,
    cache_rate: float = 1.0,
) -> CacheDataset:
    """
    Create a dataset for calcium scoring.

    Args:
        yaml_file_path (Path): The path to the yaml file containing the dataset information.
        is_training_set (bool, optional): Whether the dataset is for training. Defaults to True.
        transforms (Callable, optional): The transforms to apply to the dataset. Defaults to None.
        cache_rate (float, optional): The cache rate to use. Defaults to 1.0.

    Returns:
        CacheDataset: The dataset for calcium scoring.
    """
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    if is_training_set:
        data = yaml_data['training']
    else:
        data = yaml_data['validation']

    return CacheDataset(data=data, transform=transforms, cache_rate=cache_rate)


class CalciumScore3DDataModule(L.LightningDataModule):
    """
    Lightning DataModule for calcium scoring.

    Split samples into training and validation data.
        - Training: 80%
        - Validation: 20%

    Args:
        cascore_path_yaml (Path): Path to the yaml file containing the dataset information.
        cascore_path (Path): Path to the dataset.
        transforms (Optional[Callable]): The transforms to apply to the dataset.
        batch_size (int): The batch size to use.
    """

    def __init__(
        self,
        cascore_path_yaml: Path,
        transforms: Optional[Callable],
        batch_size: int,
        cache_rate: float = 1.0,
        num_workers: int = 4,
    ):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.cascore_path_yaml = cascore_path_yaml
        self.transforms = transforms
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = make_calcium_dataset(
                self.cascore_path_yaml,
                is_training_set=True,
                transforms=self.transforms,
                cache_rate=self.cache_rate,
            )
            self.valDataset = make_calcium_dataset(
                self.cascore_path_yaml,
                is_training_set=False,
                transforms=self.transforms,
                cache_rate=self.cache_rate,
            )
        elif stage == 'test' or stage is None:
            self.test_dataset = make_calcium_dataset(
                self.cascore_path_yaml,
                is_training_set=False,
                transforms=self.transforms,
                cache_rate=self.cache_rate,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return L.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return L.DataLoader(self.valDataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return L.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
