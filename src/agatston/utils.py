from pathlib import Path

import SimpleITK as sitk


def simpleITK_read(file_path: Path, isDataOnly: bool = False):
    """Reads an image using SimpleITK.

    Args:
        file_path (Path): path to the image file.
        isDataOnly (bool, optional): If True, only the data is returned. Defaults to False.

    Returns:
        SimpleITK.Image: The image, or the data if isDataOnly is True.
    """
    image = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(image)
    if isDataOnly:
        return data
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    return data, spacing, origin
