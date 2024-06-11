import logging

from pathlib import Path
from typing import Union

import fire
import yaml

from agatston.tools.dataTools import get_datalist_for_training
from agatston.transforms.customTransforms import BinarizeLabelsd
from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)
from tqdm import tqdm


class ImageProcessor:
    """
    Class to process medical images.
    """

    def __init__(
        self,
        setup_path: Union[str, Path],
        split_path: Union[str, Path],
        image_spacing: float = None,
        output_dir: Union[str, Path] = "/home/jacob/Developer/others/saves",
        output_ext: str = ".nii.gz",
        writer: str = "NibabelWriter",
        resample: bool = False,
        print_log: bool = True
    ):
        """
        Initialize ImageProcessor instance.

        Args:
            setup_path: Path to the setup data.
            split_path: Path to the split data.
            image_spacing: Spacing for image resizing.
            output_dir: Directory to save processed images.
            output_ext: Extension for the output images.
            writer: Image writer type.
            resample: Whether to resample images.
            print_log: Whether to print log messages.
        """
        self.setup_path = Path(setup_path)
        self.split_path = Path(split_path)
        self.image_spacing = image_spacing
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.writer = writer
        self.resample = resample
        self.print_log = print_log

        self._image_key = 'images'
        self._labels_key = 'labels'
        self._raw_plaque_key = 'raw_plaques' if self.image_spacing is None else None
        self._transformations = [
            LoadImaged(keys=[self._image_key, self._labels_key]),
            EnsureTyped(keys=[self._image_key, self._labels_key]),
            EnsureChannelFirstd(keys=[self._image_key, self._labels_key]),
            BinarizeLabelsd(keys=self._labels_key),
            Spacingd(keys=[self._image_key, self._labels_key],
                     pixdim=(self.image_spacing, self.image_spacing, self.image_spacing),
                     mode=('bilinear', 'nearest')
            ),
            Orientationd(keys=[self._image_key, self._labels_key], axcodes='RAS'),
            SaveImaged(
                keys=[self._image_key],
                output_postfix="",
                output_dir=self.output_dir,
                output_ext=self.output_ext,
                writer=self.writer,
                resample=self.resample,
                print_log=self.print_log
            ),
            ScaleIntensityRanged(keys=[self._image_key], a_min=-200, a_max=1300, b_min=0, b_max=1, clip=True),
        ] if self._raw_plaque_key is None else [
            LoadImaged(keys=[self._image_key, self._labels_key, self._raw_plaque_key]),
            EnsureTyped(keys=[self._image_key, self._labels_key, self._raw_plaque_key]),
            EnsureChannelFirstd(keys=[self._image_key, self._labels_key, self._raw_plaque_key]),
            BinarizeLabelsd(keys=[self._labels_key, self._raw_plaque_key]),
            Orientationd(keys=[self._image_key, self._labels_key, self._raw_plaque_key], axcodes='RAS'),
            SaveImaged(
                keys=[self._image_key],
                output_postfix="",
                output_dir=self.output_dir,
                output_ext=self.output_ext,
                writer=self.writer,
                resample=self.resample,
                print_log=self.print_log
            ),
            ScaleIntensityRanged(keys=[self._image_key], a_min=-200, a_max=1300, b_min=0, b_max=1, clip=True),
        ]

        self.basic_transformations = self._transformations

    def preprocess(self) -> None:
        with open(self.setup_path, 'r') as file:
            setup = yaml.safe_load(file)

        datalist = get_datalist_for_training(setup, self.split_path)

        datalist_path = Path(setup['datalist'])
        output_dir = datalist_path.parent
        output_dir.mkdir(exist_ok=True, parents=True)

        setup_number = int(Path(self.setup_path).stem.split('_')[-1])
        save_operations = []
        keys_to_process = [self._image_key, self._labels_key]
        if self.image_spacing is None:
            keys_to_process.append(self._raw_plaque_key)

        for key in keys_to_process:
            output = output_dir / f'{key}_setup_{setup_number}' if key != self._image_key else output_dir / key

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
            save_operations.append(save_operation)

        if (output_dir / self._image_key).exists():
            logging.info(
                f'Folders {self._image_key} and {self._labels_key} already exist. The images will be overwritten.')

        setup_specific_transformations = []

        if setup_number == 0:
            setup_specific_transformations += [CopyItemsd(keys=self._labels_key, names='plaques')]

        preprocessed_data_paths = {}
        transforms = Compose(self.basic_transformations + setup_specific_transformations + save_operations)

        for batch in tqdm(sum(list(datalist.values()), []), desc='Preprocessing'):
            transforms(batch)

            key = list(datalist.keys())[batch['folder']]
            id_ = Path(batch[self._image_key]).name.split('.')[0]
            sample_dict = {
                self._image_key: str(output_dir / self._image_key / f'{id_}.nii.gz'),
                self._labels_key: str(output_dir / f'{self._labels_key}_setup_{setup_number}' / f'{id_}.nii.gz'),
            }
            if self.image_spacing is None:
                sample_dict[self._raw_plaque_key] = str(
                    output_dir / f'{self._raw_plaque_key}_setup_{setup_number}' / f'{id_}.nii.gz'),

            preprocessed_data_paths[key] = preprocessed_data_paths.get(key, []) + [sample_dict]

        with open(datalist_path, 'w') as file:
            yaml.safe_dump(preprocessed_data_paths, file)


if __name__ == '__main__':
    fire.Fire(ImageProcessor)
