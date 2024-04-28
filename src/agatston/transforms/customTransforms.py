import numpy as np
import scipy.ndimage as ndi
import torch

from monai.config import KeysCollection
from monai.transforms import (
    MapTransform,
)


class BinarizeLabelsd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ) -> torch.Tensor:
        """
        Args:
            keys (KeysCollection): The key for the corresponding value.
        """
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key].set_array(np.where((d[key] == 10), 2, d[key]))
            d[key].set_array(
                np.where(
                    (d[key] == 20) | (d[key] == 30) | (d[key] == 40) | (d[key] == 50) | (d[key] == 60) | (d[key] == 70),
                    3,
                    d[key],
                )
            )
            d[key].set_array(np.where(d[key] > 3, 1, d[key]))
        return d


class RemoveSmallPlaquesd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        min_size: int = 3,
    ) -> torch.Tensor:
        """
        Args:
            keys (KeysCollection): The key for the corresponding value.
            min_size (int): The minimum size of the plaque to keep.
        """
        MapTransform.__init__(self, keys)
        self.min_size = min_size

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            connectivity = ndi.generate_binary_structure(3, 3)
            lesion_map, _ = ndi.label(d[key], connectivity)
            sizes = np.bincount(lesion_map.ravel())
            indexes_to_remove = np.where(sizes < self.min_size)[0]

            result = np.isin(lesion_map, indexes_to_remove).astype(np.uint8)
            lesion_map[result > 0] = 1
            result[result > 0] = 1

            lesion_map -= result
            d[key].set_array(lesion_map)
        return d
