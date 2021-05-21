import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *
import glob
import cv2
import os
from tqdm import tqdm
import json
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Generate evaluation scores for the detections')
    parser.add_argument('--test_frames_folder', type=str, required=False, help="Folder where test frames are stored")
    parser.add_argument('--ground_truth_folder', type=str, required=True,
                        help="Folder where ground truth txt files are stored")
    parser.add_argument('--detection_folder', type=str, required=True,
                        help="Folder where detection txt files are stored")
    parser.add_argument('--generate_visualized_frames', type=bool, default=False,
                        help="Whether to generate frames with detection and ground truth visualized")
    parser.add_argument('--visualized_frames_folder', type=str,
                        help="Folder where to save generated visualized output frames")
    parser.add_argument('--iou_threshold', type=float, default=0.5, help="Threshold for iou")

    args = parser.parse_args()
    return args


args = get_arguments()

# test_frames_folder = "~/output_dataset_combined/test/frames"
test_frames_folder = args.test_frames_folder

# ground_truth_folder = "~/output_dataset_combined/test/single_stream_pspnet_101_voc12_fine_tuned_50epochs/object_detection_metrics/groundtruths"
ground_truth_folder = args.ground_truth_folder

# detection_folder = '~/output_dataset_combined/test/single_stream_pspnet_101_voc12_fine_tuned_50epochs/object_detection_metrics/detections'
detection_folder = args.detection_folder

# generate_visualized_frames = False
generate_visualized_frames = args.generate_visualized_frames

# output_folder_with_ground_truth_detection_highlighted = "~/output_dataset_combined/test/single_stream_pspnet_101_voc12_fine_tuned_50epochs/object_detection_metrics/highlighted"
output_folder_with_ground_truth_detection_highlighted = args.visualized_frames_folder

# iou_threshold = 0.5
iou_threshold = args.iou_threshold

output_scores_json_file = os.path.realpath(os.path.join(detection_folder, '..', 'evaluation_scores.json'))


def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    # folderGT = os.path.join(currentPath, 'groundtruths')
    folderGT = ground_truth_folder
    # os.chdir(folderGT)
    files = glob.glob(os.path.join(folderGT, '*.txt'))
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = os.path.basename(f).replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            x2 = float(splitLine[3])
            y2 = float(splitLine[4])
            w = x2 - x
            h = y2 - y
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.GroundTruth,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    # folderDet = os.path.join(currentPath, 'detections')
    folderDet = detection_folder
    # os.chdir(folderDet)
    files = glob.glob(os.path.join(folderDet, "*.txt"))
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = os.path.basename(f).replace(".txt", "")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            x2 = float(splitLine[4])
            y2 = float(splitLine[5])
            w = x2 - x
            h = y2 - y
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (200, 200),
                BBType.Detected,
                confidence,
                format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes


def createImages(dictGroundTruth, dictDetected):
    """Create representative images with bounding boxes."""
    import numpy as np
    import cv2
    # Define image size
    width = 200
    height = 200
    # Loop through the dictionary with ground truth detections
    for key in dictGroundTruth:
        image = np.zeros((height, width, 3), np.uint8)
        gt_boundingboxes = dictGroundTruth[key]
        image = gt_boundingboxes.drawAllBoundingBoxes(image)
        detection_boundingboxes = dictDetected[key]
        image = detection_boundingboxes.drawAllBoundingBoxes(image)
        # Show detection and its GT
        cv2.imshow(key, image)
        cv2.waitKey()


def myCreateImage(boundingboxes):
    output_folder = output_folder_with_ground_truth_detection_highlighted
    os.makedirs(output_folder, exist_ok=True)

    frames_folder = test_frames_folder
    frame_files = glob.glob(os.path.join(frames_folder, '*'))
    frame_files.sort()

    pbar = tqdm(total=len(frame_files), desc="Saving files with bounding boxes", dynamic_ncols=True, unit="frame")
    for frame_file in frame_files:
        filename = os.path.basename(frame_file)
        filename_without_ext, file_ext = os.path.splitext(filename)

        im = cv2.imread(frame_file)
        im = boundingboxes.drawAllBoundingBoxes(im, filename_without_ext)

        cv2.imwrite(os.path.join(output_folder, filename), im)

        pbar.update()
    pbar.close()


def write_dict_to_json_file(file_path, dictionary_obj):
    with open(file_path, 'w') as fp:
        json.dump(dictionary_obj, fp)


# Read txt files containing bounding boxes (ground truth and detections)
boundingboxes = getBoundingBoxes()

if generate_visualized_frames:
    # Draw detection and ground truth on frame
    myCreateImage(boundingboxes)

# Uncomment the line below to generate images based on the bounding boxes
# createImages(dictGroundTruth, dictDetected)
# Create an evaluator object in order to obtain the metrics
evaluator = Evaluator()
##############################################################
# VOC PASCAL Metrics
##############################################################
# Plot Precision x Recall curve
# evaluator.PlotPrecisionRecallCurve(
#     boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
#     IOUThreshold=0.3,  # IOU threshold
#     method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
#     showAP=True,  # Show Average Precision in the title of the plot
#     showInterpolatedPrecision=True)  # Plot the interpolated precision curve
# Get metrics with PASCAL VOC metrics
metricsPerClass = evaluator.GetPascalVOCMetrics(
    boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=iou_threshold,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
print("Average precision values per class:\n")

# Loop through classes to obtain their metrics
for mc in metricsPerClass:
    precision = mc['precision']
    recall = mc['recall']
    total_positives = mc['total positives']
    total_TP = mc['total TP']
    total_FP = mc['total FP']
    ap = mc['AP']
    interpolated_precision = mc['interpolated precision']
    interpolated_recall = mc['interpolated recall']

    output_dict = {
        # 'precision': precision,
        # 'recall': recall,
        'total_positives': total_positives,
        'total_TP': total_TP,
        'total_FP': total_FP,
        # 'ap': ap,
    }

    # avg_precision = float(sum(precision)) / float(total_positives)
    # avg_recall = float(sum(recall)) / float(total_positives)
    avg_precision = precision[-1]
    avg_recall = recall[-1]
    f1_score = (2 * (avg_precision * avg_recall)) / (avg_precision + avg_recall)
    output_dict['precision'] = avg_precision
    output_dict['recall'] = avg_recall
    output_dict['f1_score'] = f1_score

    write_dict_to_json_file(output_scores_json_file, output_dict)

    # print('%f' % (ap))
    print(f'Total Positives (ground truths): {total_positives}')
    print(f'Total True Positives: {total_TP}')
    print(f'Total False Positives: {total_FP}')
    print(f'Precision: {avg_precision}')
    print(f'Recall: {avg_recall}')
    print(f'F1 Score: {f1_score}')

    score_details_path = os.path.join(os.path.dirname(output_scores_json_file), 'score_details.json')
    score_details = {
        'average_precision': ap,
        'precision_list': precision.tolist(),
        'recall_list': recall.tolist(),
        'interpolated_precision_list': interpolated_precision,
        'interpolated_recall_list': interpolated_recall,
    }
    with open(score_details_path, 'w') as f:
        json.dump(score_details, f)
