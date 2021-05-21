from os import path

# Folder containing test frames (absolute folder path)
frames_folder = ''
# Folder containing test ground truth frames (absolute folder path)
ground_truth_folder = ''
# Folder where to output a folder containing test results (absolute folder path)
output_path = ''
# Relative path to script for metric evaluations. This script will be used to calculate precision, recall and F1 Score
evaluation_script = 'tools/object-detection-metrics/my_evaluation_metric_code/evaluation_code.py'
evaluation_script = path.abspath(path.join(path.dirname(__file__), '..', '..', evaluation_script))
# Whether to save probabilities from network or not. Choices 0 or 1
save_probabilities = 0
