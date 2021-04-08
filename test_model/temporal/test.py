import json
import os
from glob import glob
from os import path

import cv2
import keras_segmentation
import numpy as np
from tqdm import tqdm

from test_model.temporal import config
from test_model.temporal.checkpoint_paths import checkpoints

features_folder = config.features_folder
detection_output_folder = config.detection_output_folder


def predict_segmentation_for_files(model, feature_volume_paths, output_folder):
    # make output folder if doesn't exists
    os.makedirs(output_folder, exist_ok=True)

    box_scores = []

    for feature_volume_path in tqdm(feature_volume_paths, desc="Generating predictions for feature volumes",
                                    unit='volume'):
        filename = os.path.basename(feature_volume_path)
        filename_wo_ext = os.path.splitext(filename)[0]
        output_image_path = os.path.join(output_folder, f"{filename_wo_ext}.png")

        feature_volume = np.load(feature_volume_path)

        seg_img, pr, box_score = model.predict_segmentation(
            inp=feature_volume,
            # threshold=0.4,
        )

        cv2.imwrite(output_image_path, seg_img)

        box_scores.append({
            'filename_wo_ext': filename_wo_ext,
            'score': float(box_score),
        })

    box_score_output_path = os.path.abspath(os.path.join(output_folder, '..', 'box_scores.json'))
    with open(box_score_output_path, 'w') as f:
        json.dump(box_scores, f)


def main():
    print("Running with following configurations")
    print("*************************************")
    print(f"Test features folder: {features_folder}")
    print(f"Test predictions folder: {detection_output_folder}")

    output_path = detection_output_folder

    output_folder_for_model = None

    for checkpoint in tqdm(checkpoints, desc='Processing checkpoints'):
        filename = path.basename(checkpoint)
        filename_wo_ext, file_ext = path.splitext(filename)
        weight_number = str(int(file_ext[1:]))

        if 'model' in locals():
            # del model
            model = ''
        model = keras_segmentation.predict.model_from_checkpoint_given_path(checkpoint)

        network_detections_output_folder = path.join(output_path, filename_wo_ext, weight_number, 'network_detections')

        if output_folder_for_model is None:
            output_folder_for_model = path.abspath(path.join(network_detections_output_folder, '..', '..'))

        test_feature_volumes = glob(path.join(features_folder, '*'))
        predict_segmentation_for_files(model, test_feature_volumes, network_detections_output_folder)


if __name__ == '__main__':
    main()
