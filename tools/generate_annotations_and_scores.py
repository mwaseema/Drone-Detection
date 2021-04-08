import os
from os import path

BASE_FOLDER = path.abspath(path.join(path.dirname(__file__), '..'))

# folder containing binary masks of detections
detection_folder = ''
# folder containing binary masks of ground truths
gt_folder = ''
# folder containing score files for boxes
score_boxes_folder = ''

metric_det = os.path.abspath(os.path.join(detection_folder, '..', 'metric', 'detections'))
metric_gt = os.path.abspath(os.path.join(detection_folder, '..', 'metric', 'gt'))

generate_annotation_file_path = path.join(BASE_FOLDER, 'tools/generate_annotation_files.py')
if score_boxes_folder == '':
    command = f'python {generate_annotation_file_path} --detections_folder "{detection_folder}" --detections_output_folder "{metric_det}" --ground_truth_folder "{gt_folder}" --ground_truth_output_folder "{metric_gt}"'
else:
    command = f'python {generate_annotation_file_path} --detections_folder "{detection_folder}" --detections_output_folder "{metric_det}" --ground_truth_folder "{gt_folder}" --ground_truth_output_folder "{metric_gt}" --score_boxes_folder "{score_boxes_folder}"'
os.system(command)

evaluation_code_path = path.join(BASE_FOLDER,
                                 'tools/object-detection-metrics/my_evaluation_metric_code/evaluation_code.py')
command = f'python "{evaluation_code_path}" --detection_folder "{metric_det}" --ground_truth_folder "{metric_gt}" --iou_threshold 0.5'
os.system(command)
