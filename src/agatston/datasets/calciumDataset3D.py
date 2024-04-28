from pathlib import Path
from typing import Callable, Optional

import lightning as L
import yaml

from lightning.pytorch.utilities.types import (
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)
from monai.data import CacheDataset


def makeCalciumDataset(
    yamlFilePath: Path,
    isTrainingSet: bool = True,
    transforms: Callable = None,
    cacheRate: float = 1.0,
) -> CacheDataset:
    """
    Create a dataset for calcium scoring.

    Args:
        yamlFilePath (Path): The path to the yaml file containing the dataset information.
        isTrainingSet (bool, optional): Whether the dataset is for training. Defaults to True.
        transforms (Callable, optional): The transforms to apply to the dataset. Defaults to None.
        cacheRate (float, optional): The cache rate to use. Defaults to 1.0.

    Returns:
        CacheDataset: The dataset for calcium scoring.
    """
    with open(yamlFilePath, 'r') as file:
        yamlData = yaml.safe_load(file)

    if isTrainingSet:
        data = yamlData['training']
    else:
        data = yamlData['validation']

    return CacheDataset(data=data, transform=transforms, cache_rate=cacheRate)


class CalciumScore3DDataModule(L.LightningDataModule):
    """
    Lightning DataModule for calcium scoring.

    Split samples into training and validation datasets.
        - Training: 80%
        - Validation: 20%

    Args:
        caScorePathYaml (Path): Path to the yaml file containing the dataset information.
        caScorePath (Path): Path to the dataset.
        transforms (Optional[Callable]): The transforms to apply to the dataset.
        batchSize (int): The batch size to use.
    """

    def __init__(
        self,
        caScorePathYaml: Path,
        caScorePath: Path,
        transforms: Optional[Callable],
        batchSize: int,
        cacheRate: float = 1.0,
        numWorkers: int = 4,
    ):
        super().__init__()
        self.caScorePathYaml = caScorePathYaml
        self.caScorePath = caScorePath
        self.transforms = transforms
        self.batchSize = batchSize
        self.cacheRate = cacheRate
        self.numWorkers = numWorkers

    def setup(self, stage: Optional[str] = None) -> CacheDataset:
        if stage == 'fit' or stage is None:
            self.trainDataset = makeCalciumDataset(
                self.caScorePathYaml,
                isTrainingSet=True,
                transforms=self.transforms,
                cacheRate=self.cacheRate,
            )
            self.valDataset = makeCalciumDataset(
                self.caScorePathYaml,
                isTrainingSet=False,
                transforms=self.transforms,
                cacheRate=self.cacheRate,
            )
        elif stage == 'test' or stage is None:
            self.testDataset = makeCalciumDataset(
                self.caScorePathYaml,
                isTrainingSet=False,
                transforms=self.transforms,
                cacheRate=self.cacheRate,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return L.DataLoader(self.trainDataset, batch_size=self.batchSize, shuffle=True, num_workers=self.numWorkers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return L.DataLoader(self.valDataset, batch_size=self.batchSize, shuffle=False, num_workers=self.numWorkers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return L.DataLoader(self.testDataset, batch_size=self.batchSize, shuffle=False, num_workers=self.numWorkers)
