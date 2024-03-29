###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012                   #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012                   #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

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

test_frames_folder = "/home/hec/waseem/for_evaluation_test/GT"
ground_truth_folder = "/home/hec/waseem/for_evaluation_test/text_files_for_metric/ground_truth"
detection_folder = '/home/hec/waseem/for_evaluation_test/text_files_for_metric/prediction'
output_folder_with_ground_truth_detection_highlighted = "/home/hec/waseem/for_evaluation_test/text_files_for_metric/prediction_visualized"
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
    files = glob.glob(os.path.join(folderGT,'*.txt'))
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
    IOUThreshold=0.2,  # IOU threshold
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
        #'precision': precision,
        #'recall': recall,
        'total_positives': total_positives,
        'total_TP': total_TP,
        'total_FP': total_FP,
        'ap': ap,
    }

    avg_precision = float(sum(precision)) / float(total_positives)
    avg_recall = float(sum(recall)) / float(total_positives)
    f1_score = (2 * (avg_precision * avg_recall)) / (avg_precision + avg_recall)
    output_dict['avg_precision'] = avg_precision
    output_dict['avg_recall'] = avg_recall
    output_dict['f1_score'] = f1_score

    write_dict_to_json_file(output_scores_json_file, output_dict)
    print('%f' % (ap))
