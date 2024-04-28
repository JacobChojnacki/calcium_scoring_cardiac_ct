import logging
import os
import shutil

from glob import glob
from pathlib import Path

import fire


def prepare_data(
    dataset_path: Path,
    output_path: Path,
) -> None:
    """
    Extract the images and labels from the dataset and save them in the output directory

    Args:
        dataset_path (Path): dataset path containing the images and labels
        output_path (Path): output directory where the images and labels will be saved
    """
    logging.info(f'Extracting images and labels from {dataset_path} to {output_path}')
    data_paths = {
        'images': sorted(glob(os.path.join(dataset_path, 'images', '*.nii.gz'))),
        'labels': sorted(glob(os.path.join(dataset_path, 'labels', '*.nii.gz'))),
        'raw_plaques': sorted(glob(os.path.join(dataset_path, 'raw_plaques', '*.nii.gz'))),
        'aorta': sorted(glob(os.path.join(dataset_path, 'aorta', '*.nii.gz'))),
    }
    for key in data_paths.keys():
        os.makedirs(os.path.join(output_path, key), exist_ok=True)
        for value in data_paths[key]:
            shutil.copy2(value, os.path.join(output_path, key))
    logging.info('Images and labels extracted successfully')


if __name__ == '__main__':
    fire.Fire(prepare_data)
