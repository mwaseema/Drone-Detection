import json
import os
from glob import glob
from os import path

import cv2
import keras_segmentation
import numpy as np
from tqdm import tqdm

from test_model.libs import funcs
from test_model.libs.box_utils import get_region_props
from test_model.spatial import config
from test_model.spatial.checkpoint_paths import checkpoints

test_folder = config.frames_folder
output_path = config.output_path
ground_truth_folder = config.ground_truth_folder
object_detection_metric_evaluation_script = path.abspath(
    path.join(path.dirname(__file__), '..', '..', config.evaluation_script))
save_probabilities = config.save_probabilities


def predict_segmentation_for_files(model, image_paths, output_folder):
    # make output folder if doesn't exists
    funcs.create_folders_if_not_exists(output_folder)

    for img_path in tqdm(image_paths, desc='Predicting anomaly results', unit="frame", dynamic_ncols=True):
        filename = os.path.basename(img_path)
        filename_wo_ext, _ = os.path.splitext(filename)
        output_image_path = os.path.join(output_folder, filename)

        # img_crops = funcs.get_frame_and_ground_truth_crop(img_path)
        img_crops = funcs.get_frame_and_ground_truth_crop_4splits(img_path)
        img_segmentations_1 = []
        img_segmentations_2 = []
        pred_probabilities_1 = []
        # pred_probabilities_2 = []
        for img_crop in img_crops:
            [segm_pred_1, segm_pred_2], [pred_probability_1, pred_probability_2] = model.predict_segmentation(
                inp=img_crop
            )
            # appending to the list of predictions
            img_segmentations_1.append(segm_pred_1)
            img_segmentations_2.append(segm_pred_2)

            # take foreground channel
            pred_probability_1 = pred_probability_1[:, :, 1]
            # normalize values between 0 and 1
            # pred_probability_1 -= np.min(pred_probability_1)
            # pred_probability_1 /= np.max(pred_probability_1)
            # # normalize values between 0 and 255
            # pred_probability_1 *= 255
            # pred_probability_1 = pred_probability_1.astype(np.uint8)
            # # convert it to 3 channel format
            # pred_probability_1 = cv2.cvtColor(pred_probability_1, cv2.COLOR_GRAY2BGR)
            pred_probabilities_1.append(pred_probability_1)
            # pred_probabilities_2.append(pred_probability_2)

        # merge segmentations
        # merged_mask1 = funcs.merge_segmentation_masks(img_segmentations_1)
        # merged_mask2 = funcs.merge_segmentation_masks(img_segmentations_2)
        merged_mask1 = funcs.merge_segmentation_masks_4splits(img_segmentations_1)
        merged_mask2 = funcs.merge_segmentation_masks_4splits(img_segmentations_2)

        # converting predicted region to square
        merged_mask1 = funcs.convert_segmented_area_to_bounding_box(merged_mask1)
        merged_mask2 = funcs.convert_segmented_area_to_bounding_box(merged_mask2)

        cv2.imwrite(output_image_path, merged_mask1)

        output_filename = path.basename(output_image_path)
        second_output_dirname = path.abspath(path.join(path.dirname(output_image_path), '..', 'second_detection'))
        os.makedirs(second_output_dirname, exist_ok=True)
        second_output_file_path = path.join(second_output_dirname, output_filename)
        cv2.imwrite(second_output_file_path, merged_mask2)

        if save_probabilities == 1:
            # merge probabilities
            merged_probabilities_1 = funcs.merge_segmentation_masks(pred_probabilities_1, False)

            # output to a file
            probabilities_1_folder = path.abspath(path.join(path.dirname(output_image_path), '..', 'probabilities_1'))
            os.makedirs(probabilities_1_folder, exist_ok=True)
            probabilities_1_output_path = path.join(probabilities_1_folder, filename)
            cv2.imwrite(probabilities_1_output_path, merged_probabilities_1)

        # save box probabilities
        # merge probabilities to make matrix similar to segmentation
        # merged_probabilities_1 = funcs.merge_probabilities(merged_mask1.shape, pred_probabilities_1, 50)
        merged_probabilities_1 = funcs.merge_probabilities_4splits(merged_mask1.shape, pred_probabilities_1, 50)

        merged_mask1_rps = get_region_props(merged_mask1)
        box_scores = []
        for merged_mask1_rp in merged_mask1_rps:
            y1, x1, y2, x2 = merged_mask1_rp.bbox
            probabilities_patch = merged_probabilities_1[y1:y2, x1:x2]

            box_scores.append({
                'box': {
                    'y1': y1,
                    'x1': x1,
                    'y2': y2,
                    'x2': x2,
                },
                'average_score': float(np.mean(probabilities_patch)),
                'max_score': float(np.max(probabilities_patch))
            })

        box_scores_path = path.abspath(
            path.join(path.dirname(output_image_path), '..', 'box_scores', f'{filename_wo_ext}.json'))
        os.makedirs(os.path.dirname(box_scores_path), exist_ok=True)
        with open(box_scores_path, 'w') as f:
            json.dump(box_scores, f)


def main():
    print("Running with following configurations")
    print("*************************************")
    print(f"Test frames folder: {test_folder}")
    print(f"Test ground truth folder: {ground_truth_folder}")
    print(f"Output folder: {output_path}")
    print(f"Evaluation script: {object_detection_metric_evaluation_script}\n\n")

    output_folder_for_model = None

    for checkpoint in tqdm(checkpoints, desc='Processing checkpoints'):
        filename = path.basename(checkpoint)
        filename_wo_ext, file_ext = path.splitext(filename)
        weight_number = str(int(file_ext[1:]))

        model = None
        model = keras_segmentation.predict.model_from_checkpoint_given_path(checkpoint)

        current_output_path = path.join(output_path, filename_wo_ext, weight_number, 'detections')
        os.makedirs(current_output_path, exist_ok=True)

        if output_folder_for_model is None:
            output_folder_for_model = path.abspath(path.join(current_output_path, '..', '..'))

        test_files = glob(path.join(test_folder, '*'))
        test_files.sort()
        predict_segmentation_for_files(model, test_files, current_output_path)

        metric_ground_truth = path.abspath(path.join(current_output_path, '..', 'metrics', 'ground_truth'))
        metric_detection = path.abspath(path.join(current_output_path, '..', 'metrics', 'detections'))
        # funcs.generate_annotation_text_files(ground_truth_folder, metric_ground_truth, True)
        # funcs.generate_annotation_text_files(current_output_path, metric_detection, False)
        # funcs.delete_extra_from_destination(metric_ground_truth, metric_detection)

        box_scores_folder = path.abspath(path.join(current_output_path, '..', 'box_scores'))

        generate_annotation_script = 'python '
        generate_annotation_script = generate_annotation_script + path.abspath(
            path.join(path.dirname(__file__), '..', '..', 'tools/generate_annotation_files.py'))
        generate_annotation_script += f' --detections_folder "{current_output_path}"'
        generate_annotation_script += f' --detections_output_folder "{metric_detection}"'
        generate_annotation_script += f' --ground_truth_folder "{ground_truth_folder}"'
        generate_annotation_script += f' --ground_truth_output_folder "{metric_ground_truth}"'
        generate_annotation_script += f' --score_boxes_folder "{box_scores_folder}"'
        os.system(generate_annotation_script)

        os.system(
            rf'python "{object_detection_metric_evaluation_script}" --detection_folder "{metric_detection}" --ground_truth_folder "{metric_ground_truth}" --iou_threshold 0.5')

    # get all the scores files and store in single file after reading
    with open(path.join(output_folder_for_model, 'all_scores.txt'), 'w') as wf:
        score_files = glob(path.join(output_folder_for_model, '*', 'metrics', 'evaluation_scores.json'))
        for sf_path in score_files:
            epoch_number = path.basename(path.abspath(path.join(path.dirname(sf_path), '..')))

            with open(sf_path, 'r') as rf:
                read = rf.read()

                wf.write(epoch_number)
                wf.write("\n")
                wf.write(read)
                wf.write("\n\n")


if __name__ == '__main__':
    main()
