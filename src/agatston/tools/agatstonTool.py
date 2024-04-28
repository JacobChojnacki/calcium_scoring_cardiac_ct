from pathlib import Path

import numpy as np
import scipy

from agatston.utils import simpleITK_read


def density_factor(image: np.ndarray, mask: float) -> float:
    """
    The calculation is based on the weighted density score given
    to the highest attenuation value (HU) multiplied by the area of the calcification speck:
    * 130-199 HU: 1
    * 200-299 HU: 2
    * 300-399 HU: 3
    * 400+ HU: 4
    Args:
        image (np.ndarray): The image containing the object.
        mask (float): The object to compute the density factor of.

    Returns:
        float: The density factor of the object in the image.
    """
    max_hu = np.max(image)
    if 130 <= max_hu < 200:
        return 1 * mask
    elif 200 <= max_hu < 300:
        return 2 * mask
    elif 300 <= max_hu < 400:
        return 3 * mask
    elif max_hu >= 400:
        return 4 * mask


def agatston_score(
    image_path: Path = None,
    mask_path: Path = None,
    isCalciumVolume: bool = False,
):
    """Calculates the Agatston score based on DOI: 10.1118/1.4945696.

    Args:
        image_path (Path, optional): _description_. Defaults to None.
        mask_path (Path, optional): _description_. Defaults to None.
        isCalciumVolume (bool, optional): _description_. Defaults to False.
    """
    agatston_score = 0

    if isCalciumVolume:
        mask, mask_spacing, mask_origin = simpleITK_read(mask_path)
        mask[mask > 1] = 1
        mask = mask.astype(np.float32)
        calcium_volume = 0
    else:
        image, image_spacing, image_origin = simpleITK_read(image_path)
        mask, mask_spacing, mask_origin = simpleITK_read(mask_path)
        mask[mask > 1] = 1

    mask_volume = np.prod(mask_spacing)

    for slice in range(mask.shape[0]):
        if isCalciumVolume:
            mask_slice = mask[slice]
        else:
            mask_slice = mask[slice]
            image_slice = image[slice]

        mask_slice_label, num_slice_labels = scipy.ndimage.label(mask_slice, structure=np.ones((3, 3)))

        if num_slice_labels > 0:
            for label_count in range(1, num_slice_labels + 1):
                label = np.zeros_like(mask_slice.shape)
                label[mask_slice_label == label_count] = 1
                if isCalciumVolume:
                    calcium_volume += np.sum(label) * mask_volume
                else:
                    cluster_volume = image_slice * label

                    if np.sum(cluster_volume) > 3:
                        continue

                    roi_area = np.sum(label)
                    agatston_score += round(density_factor(cluster_volume, roi_area), 3)
    if isCalciumVolume:
        return calcium_volume
    return agatston_score
