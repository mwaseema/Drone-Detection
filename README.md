# Drone-Detection

**Paper title** \
Dogfight: Detecting Drones from Drones Videos \
Accepted for CVPR 2021\
Preprint available at [arxiv](https://arxiv.org/abs/2103.17242)

# Setup environment

`environment.yml` file contains the environment and package configuration information with proper versions. For our
setup, we’ve used CUDA 9.0.176 and would recommend to use the same as it will support all the packages and their
versions listed in the environment.yml file.

Make sure you’ve conda installed in your system and is accessible before running following command to create a new
environment named “drones_detections”\
`conda env create -f environment.yml`

After making a new environment, activate it using the following command:\
`conda activate drones_detections`

Here `drones_detections` is the name of the environment. After the activation of the environment, run the following
commands to install a dependency package:

```
git clone https://github.com/mwaseema/image-segmentation-keras-implementation
cd image-segmentation-keras-implementation
make
```

**Main dependencies:**

Keras = 2.0.8\
OpenCV = 3.4.2\
Tensorflow = 1.12.0

**Other dependencies:**

CUDA = 9.0.176\
Nvidia Driver = 384.130

**Tested on:**

Ubuntu = 16.04\
Ubuntu = 18.04

# Spatial (2D) stage

## Generate image crops

We had high resolution images with really small objects because of which we’ve made overlapping 9 crops of them.
Following file can be used to generate crops from images:\
`tools/image_crops/generate_crops_of_images_by_dividing.py`

This file contains a `GenerateCrops` class which has variables for input and output data, set appropriate values for
those variables and run the code. Implementation for getting 9 and 4 small patches from given images is available in the
code available within “tools/image_crops” directory.

## Training

Set values for config variables available in `train/spatial/config.py` file and start training of the model by executing
the `train/spatial/train.py` file.

## Testing

Copy absolute path of the weight(s) for which you want to perform evaluation on your testing data. Paste absolute path
of training weights in `test_model/spatial/checkpoint_paths.py` file as array element(s). Add configuration values
in `test_model/spatial/config.py` file for available variables and run `test_model/spatial/test_and_score.py` file for
running evaluations on the testing data.

# Temporal (3D) stage

## Motion boundaries

We’ve obtained motion boundaries using optical flow for getting good candidate regions. Following code file can be used
to generate and save motion boundaries for the given videos:\
`tools/optical_flow_motion_boundaries.py`

Provide a path to the folder containing the videos and where to output the motion boundaries before running the code
file.

We’ve used the above code for generating motion boundaries for NPS dataset but generating motion boundaries for
FL-drones dataset was challenging as background motion dominated the drones. To tackle this problem we’ve used motion
stabilization before generating motion boundaries. Following code file generates motion boundaries after stabilization
of the frames:\
`tools/optical_flow_motion_boundaries_with_stabilization.py`

Values for some variables are required before executing the script.

### Motion boundaries edges

Motion boundaries generated after stabilizations had high values at borders which were removed by using following code:\
`tools/remove_motion_boundary_edges.py`

## Motion boundaries dilation

We’ve used the following code to dilate the thin motion boundaries to get candidate regions which cover drones well:\
`tools/binary_mask_dilation.py`

## Remove irrelevant candidate regions

After dilation of motion boundaries some candidate regions are represented by big motion boundaries which cannot
represent drones correctly. Such regions are removed by using the following code file. Threshold for the small box in
the file is kept at 0 to make sure only large boxes are removed:\
`tools/remove_small_big_boxes.py`

## CRF on the candidate boxes

We are using CRF to make sure the candidate boxes obtained from motion boundaries are tightly packed around the drones.\
`tools/crf/crf_on_labels.py`

This code file accepts following parameters:\
`--frames_folder`: This is where video frames in png format are saved\
`--labels_mask_folder`: Set this to the folder containing the binary masks obtained after removing irrelevant large
boxes from candidate regions\
`--output_folder`: Set this to the folder where you want to output binary masks after applying CRF\
`--save_boxes_as_json`: Set this to true to save boxes after applying CRF as JSON files.

After executing above code, following code file can be used to convert boxes which were obtained in JSON format to a
custom format which is used in further next step:\
`tools/boxes_list_to_patch_information_json.py`

## Generating cuboids

We’ve used fixed sized cuboids for NPS dataset and Multi scaled cuboids for FL-Drones dataset.

### Fixed sized cuboids

Following script can be used to generate fixed sized cuboids (This is used for NPS Drones dataset):\
`tools/cuboids/volumes_with_tracking_of_generated_boxes_with_stabilization.py`

This script will use patch information json files generated in the previous step.

### Multi sized cuboids

Following script can be used to generate multi sized cuboids (This is used for FL-Drones dataset):\
`tools/cuboids/multi_scale_volumes_with_tracking_and_stabilization_using_masks.py`

In case of multi sized cuboids; ground truths, 2d detections etc are transformed along with the frames. This was to
calculate scores without transforming the detections back using inverse matrices.

If you experience any problem related to stabilization, try lowering the max number of corner points (available at line 218) used for video
stabilization.

## I3D features

We’ve used kinetics I3D’s from deepmind with pretrained weights for extracting features from generated cuboids.
Repository for I3D is available at:\
[deepmind/kinetics-i3d: Convolutional neural network model for video classification trained on the Kinetics dataset.](https://github.com/deepmind/kinetics-i3d)

Instead of getting output from the last layer, we’ve obtained features from the middle layer which is of dimensions
1x2x14x14x480. These features are averaged over the 2nd axis and then reshaped into 14x14x480 before passing them
through our proposed temporal pipeline. Also, we only used the RGB stream of I3D instead of using two streamed networks.

## Training

For temporal stage training, set values in `train/temporal/config.py` and start training
using `train/temporal/train_model.py`

## Testing

Copy absolute path of the weight(s) for which you want to perform evaluation on your testing data and paste them
in `test_model/temporal/checkpoint_paths.py` file as array element(s). Add configuration values
in `test_model/temporal/config.py` file for available variables and run `test_model/temporal/test.py` file for running
evaluations on the testing data.

### NMS

After generating results from the temporal stage using the I3D features, pass the predictions through NMS stage.\
`tools/nms/nms_generated_boxes.py`

### Results generation

Following code can be used for generating results:\
`test_model/temporal/results_generation.py`

Results can also be computed using the code available in following file:\
`tools/generate_annotations_and_scores.py`

### Temporal consistency

We’ve used temporal consistency to remove any noisy false positives from the detections. Code for this is available in
following file:\
`tools/video_tubes/remove_noisy_false_positives_by_tracking.py`

If you find any bug, or have some questions, please contact M. Waseem Ashraf (mohammadwaseem043 [at] gmail.com) and
Waqas Sultani (waqas5163 [at] gmail.com)

# Annotations

Annotations for both, NPS-Drones and FL-Drones dataset are available
in [annotations folder](https://github.com/mwaseema/Drone-Detection/tree/main/annotations).

## Format

Every video has corresponding annotation file containing bounding box coordinates for every frame. A single bounding box
in a frame is represented by a line in the annotation file as follows:

```
frame number, number of bounding boxes, x1, y1, x2, y2
```

If a frame contains multiple bounding boxes, they are represented by a line in the annotation file as follows:

```
frame number, number of bounding boxes, box_1_x1, box_1_y1, box_1_x2, box_1_y2, box_2_x1, box_2_y1, box_2_x2, box_2_y2, ...
```

First frame is represented by frame number 0, second frame is represented by frame number 1 and so on...

# Citation

```
@article{ashraf2021dogfight,
  title={Dogfight: Detecting Drones from Drones Videos},
  author={Ashraf, Muhammad Waseem and Sultani, Waqas and Shah, Mubarak},
  journal={arXiv preprint arXiv:2103.17242},
  year={2021}
}
```
