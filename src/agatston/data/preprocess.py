import logging

from pathlib import Path

import fire
import yaml

from agatston.tools.dataTools import get_datalist_for_training
from agatston.transforms.customTransforms import BinarizeLabelsd
from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from tqdm import tqdm


def preprocess(setup_path: Path, split_path: Path, image_spacing: float = 0.6) -> None:
    """
    Preprocess the images and labels according to the setup file and the split file and save them in the output
    directory defined in the setup file.

    Args:
        setup_path (Path): setup file containing the setup parameters like "class_names" and "labels"
        split_path (Path): split file containing the split
        image_spacing (float, optional): image spacing. Defaults to 0.3.
    """
    with open(setup_path, 'r') as file:
        setup = yaml.safe_load(file)

    datalist = get_datalist_for_training(setup, split_path)

    datalist_path = Path(setup['datalist'])
    output_dir = datalist_path.parent
    datalist_path.parent.mkdir(exist_ok=True, parents=True)

    setup_number = int(Path(setup_path).name.split('.')[0].split('_')[-1])

    # split_type = list(datalist.keys())[0]
    # [name for name in datalist[split_type][0].keys() if name not in ['folder', 'images']]

    image_key = 'images'
    labels_key = 'labels'
    # raw_plaques_key = "raw_plaques"
    basic_transformations = [
        LoadImaged(keys=[image_key, labels_key]),
        EnsureChannelFirstd(keys=[image_key, labels_key]),
        BinarizeLabelsd(keys=labels_key),
        Spacingd(
            keys=[image_key, labels_key],
            pixdim=(image_spacing, image_spacing, image_spacing),
            mode=('bilinear', 'nearest'),
        ),
        Orientationd(keys=[image_key, labels_key], axcodes='RAS'),
        ScaleIntensityRanged(keys=[image_key], a_min=-200, a_max=1300, b_min=0, b_max=1, clip=True),
    ]

    save_operations = []

    if (output_dir / image_key).exists():
        logging.info(f'Folders {image_key} and {labels_key} already exist. The images will be overwritten.')

    for key in [image_key, labels_key]:
        output = output_dir / key if key == image_key else output_dir / f'{key}_setup_{setup_number}'

        save_operation = SaveImaged(
            keys=key,
            output_postfix='',
            output_dir=output,
            output_ext='.nii.gz',
            writer='NibabelWriter',
            separate_folder=False,
            resample=False,
            print_log=True,
        )
        save_operations += [save_operation]

    setup_specific_transformations = []

    if setup_number == 0:
        setup_specific_transformations += [CopyItemsd(keys=labels_key, names="plaques")]

    preprocessed_data_paths = {}
    transforms = Compose(basic_transformations + setup_specific_transformations + save_operations)

    for batch in tqdm(sum(list(datalist.values()), []), desc='Preprocessing'):
        transforms(batch)

        key = list(datalist.keys())[batch['folder']]
        id_ = Path(batch[image_key]).name.split('.')[0]

        sample_dict = {
            image_key: str(output_dir / image_key / f'{id_}.nii.gz'),
            labels_key: str(output_dir / f'{labels_key}_setup_{setup_number}' / f'{id_}.nii.gz'),
        }
        preprocessed_data_paths[key] = preprocessed_data_paths.get(key, []) + [sample_dict]

    with open(datalist_path, 'w') as file:
        yaml.safe_dump(preprocessed_data_paths, file)


if __name__ == '__main__':
    fire.Fire(preprocess)
