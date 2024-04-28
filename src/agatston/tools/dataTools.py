from pathlib import Path
from typing import Any

import yaml


def create_training_datalist(setup: dict[str, Any], splits: dict[str, list[str]]) -> dict[str, list[dict[str, str]]]:
    """_summary_

    Args:
        setup (dict[str, Any]): dictionary containing the setup parameters like "class_names" and "labels"
        splits (dict[str, list[str]]): a dictionary containing the split names as keys and the list of filenames as
                                       values

    Returns:
        dict[str, list[dict[str, str]]]: A dictionary where keys are split names and values are lists of dictionaries
                                         containing the filename and label
    """
    datalist = {}
    labels = dict(zip(setup['class_names'][1:], setup['labels']))
    raw_plaques = dict(zip(setup['class_names'][2:], setup['raw_plaques']))
    aorta = dict(zip(setup['class_names'][3:], setup['aorta']))
    for index, key in enumerate(splits.keys()):
        datalist[key] = [
            {
                **{'folder': index, 'images': f"{setup['images']}/{id_}.nii.gz"},
                **{name: f'{path}/{id_}.nii.gz' for (name, path) in labels.items()},
                **{name: f'{path}/{id_}.nii.gz' for (name, path) in raw_plaques.items()},
                **{name: f'{path}/{id_}.nii.gz' for (name, path) in aorta.items()},
            }
            for id_ in splits[key]
        ]
    return datalist


def get_datalist_for_training(setup: dict[str, Any], split_path: Path) -> dict[str, list[dict[str, str]]]:
    """Load the split files and create the datalist for training

    Args:
        setup (dict[str, Any]): A dictionary containing the setup parameters like "class_names" and "labels"
        split_path (Path): The path to the yaml file containing the split

    Returns:
        dict[str, list[dict[str, str]]]: A dictionary where keys are split names and values are lists of dictionaries
        containing the filename and label
    """
    with open(split_path, 'r') as file:
        splits = yaml.safe_load(file)

    assert all(name in splits.keys() for name in ['training', 'valid'])
    assert 'images' in setup.keys() and 'labels' in setup.keys()

    return create_training_datalist(setup, splits)
