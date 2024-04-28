from pathlib import Path

import SimpleITK as sitk


def simple_itk_read(file_path: Path, is_data_only: bool = False):
    """Reads an image using SimpleITK.

    Args:
        file_path (Path): path to the image file.
        is_data_only (bool, optional): If True, only the data is returned. Defaults to False.

    Returns:
        SimpleITK.Image: The image, or the data if isDataOnly is True.
    """
    image = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(image)
    if is_data_only:
        return data
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    return data, spacing, origin
