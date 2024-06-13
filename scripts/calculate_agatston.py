import os

from glob import glob

import fire
import pandas as pd
import tqdm

from agatston.tools import agatston_tool


def main(
    image_folder_path='/home/jacob/Desktop/the_same_size/images',
    inference_folder_path='/home/jacob/Desktop/the_same_size/postprocess_smaller_data',
    label_folder_path='/home/jacob/Desktop/the_same_size/labels',
):
    """Validate model results. Calculate agatston for inference and compare it with ground truth.
    Results are stored in csv file with the following columns:
    * agatston_inference
    * agatston_labels
    * agatston_difference
    * plaque_volume_inference
    * plaque_volume_labels
    * plaque_volume_differece

    Args:
        image_folder_path (str, optional): Path to the image folder.
        inference_folder_path (str, optional): Path to the inference results folder.
        label_folder_path (str, optional): Path to the ground truth labels folder.
    """
    # Collect file paths
    image_files = sorted(glob(os.path.join(image_folder_path, '*')))
    inference_files = sorted(glob(os.path.join(inference_folder_path, '*')))
    label_files = sorted(glob(os.path.join(label_folder_path, '*')))

    # Create DataFrame
    df_results = pd.DataFrame(
        {'images_path': image_files, 'inference_path': inference_files, 'labels_path': label_files}
    )

    # Initialize columns for the results
    df_results['agatston_inference'] = None
    df_results['agatston_labels'] = None
    df_results['agatston_difference'] = None
    df_results['plaque_volume_inference'] = None
    df_results['plaque_volume_labels'] = None
    df_results['plaque_volume_difference'] = None
    df_results['patient'] = None

    for index, row in tqdm.tqdm(df_results.iterrows(), total=df_results.shape[0]):
        image_path = row['images_path']
        inference_path = row['inference_path']
        label_path = row['labels_path']

        # Calculate Agatston scores
        agatston_inference = agatston_tool.agatston_score(image_path=image_path, mask_path=inference_path)
        agatston_labels = agatston_tool.agatston_score(image_path=image_path, mask_path=label_path)

        # Calculate plaque volumes
        plaque_volume_inference = agatston_tool.agatston_score(mask_path=inference_path, is_calcium_volume=True)
        plaque_volume_labels = agatston_tool.agatston_score(mask_path=label_path, is_calcium_volume=True)

        # Extract patient identifier
        patient_id = os.path.basename(inference_path)[:6]

        # Assign calculated values to DataFrame
        df_results.at[index, 'agatston_inference'] = agatston_inference
        df_results.at[index, 'agatston_labels'] = agatston_labels
        df_results.at[index, 'plaque_volume_inference'] = plaque_volume_inference
        df_results.at[index, 'plaque_volume_labels'] = plaque_volume_labels
        df_results.at[index, 'patient'] = patient_id

    # Calculate differences
    df_results['agatston_difference'] = df_results['agatston_labels'] - df_results['agatston_inference']
    df_results['plaque_volume_difference'] = df_results['plaque_volume_labels'] - df_results['plaque_volume_inference']

    # Set patient as index and drop unnecessary columns
    df_results.set_index('patient', inplace=True)
    df_results.drop(columns=['images_path', 'inference_path', 'labels_path'], inplace=True)

    # Print results
    print(df_results)
    print(
        '_______________________________________________________________________________________________________________'
    )
    print('Agatston')
    print('mean: ', df_results['agatston_difference'].mean())
    print('std: ', df_results['agatston_difference'].std())
    print('Plaque Volume')
    print('mean: ', df_results['plaque_volume_difference'].mean())
    print('std: ', df_results['plaque_volume_difference'].std())

    # Save results to CSV
    df_results.to_csv('model_results.csv')


if __name__ == '__main__':
    fire.Fire(main)
