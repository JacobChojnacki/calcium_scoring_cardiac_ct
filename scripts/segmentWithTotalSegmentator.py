import logging
import os
import subprocess

from glob import glob

import fire


def segment_with_total_segmentator(
    dataset_path: str,
    output_path: str,
    total_segmentator_path: str,
    is_first_launch: bool = True,
) -> None:
    """
    Segment the images using the TotalSegmentator model and save the segmentations in the output directory
    specified in the setup file. The model will be downloaded if it is the first launch.

    Args:
        dataset_path (str): dataset path containing the images
        output_path (str): output directory where the segmentations will be saved
        total_segmentator_path (str): path to the TotalSegmentator weights
        isFirstLaunch (bool, optional): flag to download the TotalSegmentator weights. Defaults to True.
    """
    if is_first_launch:
        subprocess.run(f'totalseg_import_weights -i {total_segmentator_path}', shell=True)
    logging.info(f'Segmenting images from {dataset_path} to {output_path}')
    for image in sorted(glob(os.path.join(dataset_path, '*.nii.gz'))):
        fileName = os.path.basename(image.split('.')[0])
        subprocess.run(f'TotalSegmentator -i {image} -o {os.path.join(output_path, fileName)} --fast', shell=True)
        subprocess.run(
            f'TotalSegmentator -i {image} -o {os.path.join(output_path, fileName)} -ta heartchambers_test', shell=True
        )  # noqa: E501
    logging.info('Images segmented successfully')


if __name__ == '__main__':
    fire.Fire(segment_with_total_segmentator)
