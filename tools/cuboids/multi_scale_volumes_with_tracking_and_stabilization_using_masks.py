import json
import multiprocessing
import os
from glob import glob
from typing import List, Tuple

import cv2
import cv2 as cv
import numpy as np
from tqdm import tqdm

from tools.crf.IoU_test import get_region_props
from tools.crf.crf_on_labels import get_safe_extended_pixels
from tools.motion_stablization.stabalize_motion import smooth
from tools.video_tubes.remove_noisy_false_positives_by_tracking import get_video_wise_list

# folder containing frames of the videos
frames_folder = ''
# masks in this folder will be used for processing (tracking)
# set this variable to the folder containing binary masks generated by CRF
mask_folder = ''
# set following variable to the path of ground truth binary masks
# these will be transformed and patches of these will be saved
mask2_folder = ''
# set path to spatial prediction masks
detection_2d_mask_folder = ''

# where to store the output files
# i.e. ~/cuboids
output_folder = ''
output_filename_prefix = ''

# ~/cuboids/__gt_transformed
gt_output_folder = ''
# ~/cuboids/__2d_detections_transformed
mask_2d_output_folder = ''
frame_output_folder = ''

volume_dimensions = 224
foreground_pixel_value = 1
extra_pixels = 50

frames_in_volume = 8
middle_frame_number = frames_in_volume // 2


def extract_small_boxes(boxes: list):
    new_boxes = []
    for box in boxes:
        new_boxes.append(box['box'])
    return new_boxes


def extend_height_width(height, width, cy, cx, height_dimension, width_dimension):
    either_side_width = width_dimension / 2
    either_side_height = height_dimension / 2

    y1 = cy - either_side_height
    y2 = cy + either_side_height

    x1 = cx - either_side_width
    x2 = cx + either_side_width

    y1 = y1 if y1 > 0 else 0
    x1 = x1 if x1 > 0 else 0

    y2 = y2 if y2 < height else height
    x2 = x2 if x2 < width else width

    return int(y1), int(x1), int(y2), int(x2)


def get_patches_by_tracker_extend_version(tracker, frames_list: List[np.ndarray], box_height, box_width):
    patches = []
    for frame in frames_list:
        ret, tracker_box = tracker.update(frame)
        x1, y1, w, h = tracker_box
        cy = y1 + (h / 2)
        cx = x1 + (w / 2)

        p_y1, p_x1, p_y2, p_x2 = extend_height_width(frame.shape[0], frame.shape[1], cy, cx, box_height, box_width)

        # safe_extended_pixels = get_safe_extended_pixels(frame.shape[0], frame.shape[1], extra_pixels, y1, x1, y2, x2)
        # p_y1, p_x1, p_y2, p_x2 = [int(_) for _ in safe_extended_pixels]

        patches.append(frame[p_y1:p_y2, p_x1:p_x2])
    return patches


def get_tracked_volumes_boxes(frames_list: List[np.ndarray], boxes: List[Tuple[int, int, int, int]],
                              key_frame_number: int):
    key_frame = frames_list[key_frame_number]

    # from frame 0 to less than key frame
    previous_frames = frames_list[0:key_frame_number]
    # reverse frames
    previous_frames = previous_frames[-1::-1]

    # from keyframe+1 to end
    next_frames = frames_list[key_frame_number + 1:]

    final_volumes = []
    # get boxes for key frames
    for box in boxes:
        y1, x1, y2, x2 = box
        w = x2 - x1
        h = y2 - y1

        if w < 10:
            w = 10

        if h < 10:
            h = 10

        tracker = cv.TrackerCSRT_create()
        try:
            tracker.init(key_frame, (x1, y1, w, h))
        except:
            print('Box ignored because tracker could not initialize')
            continue

        key_frame_patch_location = get_safe_extended_pixels(key_frame.shape[0], key_frame.shape[1], extra_pixels, y1,
                                                            x1, y2, x2)
        extended_height = key_frame_patch_location[2] - key_frame_patch_location[0]
        extended_width = key_frame_patch_location[3] - key_frame_patch_location[1]
        key_frame_patch = key_frame[key_frame_patch_location[0]:key_frame_patch_location[2],
                          key_frame_patch_location[1]:key_frame_patch_location[3]]

        previous_patches = get_patches_by_tracker_extend_version(tracker, previous_frames, extended_height,
                                                                 extended_width)
        previous_patches = previous_patches[-1::-1]

        next_patches = get_patches_by_tracker_extend_version(tracker, next_frames, extended_height, extended_width)

        final_volume = []
        final_volume.extend(previous_patches)
        final_volume.append(key_frame_patch)
        final_volume.extend(next_patches)

        final_volumes.append({
            'location': key_frame_patch_location,
            'volume': final_volume
        })
    return final_volumes


def get_tracked_volumes(frames_list: List[np.ndarray], masks_list: List[np.ndarray], key_frame_number: int,
                        detection_masks_list: List[np.ndarray] = None):
    key_frame = frames_list[key_frame_number]

    # from frame 0 to less than key frame
    previous_frames = frames_list[0:key_frame_number]
    # reverse frames
    previous_frames = previous_frames[-1::-1]

    # from keyframe+1 to end
    next_frames = frames_list[key_frame_number + 1:]

    final_volumes = []
    # get boxes for key frames
    boxes = get_region_props(masks_list[key_frame_number])
    if detection_masks_list is not None:
        boxes.extend(get_region_props(detection_masks_list[key_frame_number]))
    for box in boxes:
        y1, x1, y2, x2 = box.bbox
        w = x2 - x1
        h = y2 - y1

        if w < 10:
            w = 10

        if h < 10:
            h = 10

        tracker = cv.TrackerCSRT_create()
        try:
            tracker.init(key_frame, (x1, y1, w, h))
        except:
            print('Box ignored because tracker could not initialize')
            continue

        key_frame_patch_location = get_safe_extended_pixels(key_frame.shape[0], key_frame.shape[1], extra_pixels, y1,
                                                            x1, y2, x2)
        extended_height = key_frame_patch_location[2] - key_frame_patch_location[0]
        extended_width = key_frame_patch_location[3] - key_frame_patch_location[1]
        key_frame_patch = key_frame[key_frame_patch_location[0]:key_frame_patch_location[2],
                          key_frame_patch_location[1]:key_frame_patch_location[3]]

        previous_patches = get_patches_by_tracker_extend_version(tracker, previous_frames, extended_height,
                                                                 extended_width)
        previous_patches = previous_patches[-1::-1]

        next_patches = get_patches_by_tracker_extend_version(tracker, next_frames, extended_height, extended_width)

        final_volume = []
        final_volume.extend(previous_patches)
        final_volume.append(key_frame_patch)
        final_volume.extend(next_patches)

        final_volumes.append({
            'location': key_frame_patch_location,
            'volume': final_volume
        })
    return final_volumes


def get_motion_stabilization(frames: List[np.ndarray], masks: List[np.ndarray] = None,
                             gt_masks: List[np.ndarray] = None, detection_2d_masks=None):
    n_frames = len(frames)
    height, width = frames[0].shape[0:2]

    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(len(frames) - 1):
        prev = frames[i]
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

        curr = frames[i + 1]
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Track feature points
        # status = 1. if flow points are found
        # err if flow was not find the error is not defined
        # curr_pts = calculated new positions of input features in the second image
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        assert prev_pts.shape == curr_pts.shape

        # fullAffine= FAlse will set the degree of freedom to only 5 i.e translation, rotation and scaling
        # try fullAffine = True
        m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False)

        assert m is not None, "m shouldn't be none"

        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]

    # Find the cumulative sum of tranform matrix for each dx,dy and da
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    stabilized_frames = []
    stabilized_masks = []
    gt_stabilized_masks = []
    inverse_matrices = []
    detection_2d_stabilized_masks = []

    # Write n_frames-1 transformed frames
    for i in range(len(frames) - 1):
        frame = frames[i]

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        stabilized_frames.append(frame_stabilized)

        if masks is not None:
            mask = masks[i]
            mask_stabilized = cv2.warpAffine(mask, m, (width, height))
            stabilized_masks.append(mask_stabilized)

        if gt_masks is not None:
            gt_mask = gt_masks[i]
            gt_mask_stabilized = cv2.warpAffine(gt_mask, m, (width, height))
            gt_stabilized_masks.append(gt_mask_stabilized)

        if detection_2d_masks is not None:
            detection_2d_mask = detection_2d_masks[i]
            detection_2d_mask_stabilized = cv2.warpAffine(detection_2d_mask, m, (width, height))
            detection_2d_stabilized_masks.append(detection_2d_mask_stabilized)

        m_inv = np.zeros((2, 3), np.float32)
        m_inv[0, 0] = np.cos(da)
        m_inv[0, 1] = np.sin(da)
        m_inv[1, 0] = -np.sin(da)
        m_inv[1, 1] = np.cos(da)
        m_inv[0, 2] = -dx * np.cos(da) - dy * np.sin(da)
        m_inv[1, 2] = -dy * np.cos(da) + dx * np.sin(da)
        inverse_matrices.append(m_inv)
        # frame_inv = frame_stabilized.copy()
        # frame_inv = cv2.warpAffine(frame_inv, m_inv, (width, height))

    return stabilized_frames, inverse_matrices, stabilized_masks, gt_stabilized_masks, detection_2d_stabilized_masks


def main_process(params):
    video_frames = params['video_frames']
    frame_number = params['frame_number']
    output_volumes_folder = params['output_volumes_folder']
    output_masks_folder = params['output_masks_folder']
    output_patch_information_folder = params['output_patch_information_folder']

    middle_frame_path = video_frames[frame_number + middle_frame_number]

    filename = os.path.basename(middle_frame_path)
    filename_wo_ext = os.path.splitext(filename)[0]

    middle_mask_path = os.path.join(mask_folder, f'{filename_wo_ext}.png')
    if os.path.exists(middle_mask_path):
        middle_mask = cv.imread(middle_mask_path)
        middle_gt_mask = cv.imread(os.path.join(mask2_folder, f'{filename_wo_ext}.png'))
        middle_detections_2d_mask = cv.imread(os.path.join(detection_2d_mask_folder, f'{filename_wo_ext}.png'))

        frames_list = []
        for i in range(frame_number, frame_number + frames_in_volume + 1, 1):
            f_path = video_frames[i]
            frames_list.append(cv.imread(f_path))

        __masks_list = []
        __gt_list = []
        __detection_2d_masks_list = []
        for i in range(len(frames_list)):
            __masks_list.append(middle_mask)
            __gt_list.append(middle_gt_mask)
            __detection_2d_masks_list.append(middle_detections_2d_mask)

        try:
            frames_list, inverse_matrices, stabilized_masks, gt_stabilized_masks, stabilized_detection_2d_masks = get_motion_stabilization(
                frames_list, __masks_list, __gt_list, __detection_2d_masks_list)
        except Exception as __exp:
            print(f"Stabilization failed for {filename}")
            return ''

        # set stabilized masks to empty masks if there is no foreground in the gt of the mask
        c_mask = __masks_list[middle_frame_number]
        c_gt_mask = __gt_list[middle_frame_number]
        c_det_mask = stabilized_detection_2d_masks[middle_frame_number]
        if np.max(c_gt_mask) == 0:
            stabilized_masks[middle_frame_number] = c_mask
            gt_stabilized_masks[middle_frame_number] = c_gt_mask
            stabilized_detection_2d_masks[middle_frame_number] = c_det_mask

        # thresholding interpolated values
        c_mask = stabilized_masks[middle_frame_number]
        c_gt_mask = gt_stabilized_masks[middle_frame_number]
        c_det_mask = stabilized_detection_2d_masks[middle_frame_number]
        c_mask[c_mask < 30] = 0
        c_mask[c_mask >= 30] = 255
        c_gt_mask[c_gt_mask < 30] = 0
        c_gt_mask[c_gt_mask >= 30] = 255
        c_det_mask[c_det_mask < 30] = 0
        c_det_mask[c_det_mask >= 30] = 255
        stabilized_masks[middle_frame_number] = c_mask
        gt_stabilized_masks[middle_frame_number] = c_gt_mask
        stabilized_detection_2d_masks[middle_frame_number] = c_det_mask

        tracked_volumes = get_tracked_volumes(frames_list, stabilized_masks, middle_frame_number,
                                              stabilized_detection_2d_masks)
        for volume_number, tracked_volume in enumerate(tracked_volumes):
            tvl_y1, tvl_x1, tvl_y2, tvl_x2 = tracked_volume['location']

            volume = tracked_volume['volume']
            for i in range(len(volume)):
                volume[i] = cv.resize(volume[i], (volume_dimensions, volume_dimensions))
            volume = np.array(volume)

            output_filename = f'{output_filename_prefix}{filename_wo_ext}_{str(volume_number).zfill(3)}'
            volume_output_path = os.path.join(output_volumes_folder, output_filename + '.npy')
            np.save(volume_output_path, volume)

            mask_output_path = os.path.join(output_masks_folder, output_filename + '.png')
            # middle_mask_patch = middle_mask[tvl_y1:tvl_y2, tvl_x1:tvl_x2]
            middle_mask_patch = __gt_list[middle_frame_number][tvl_y1:tvl_y2, tvl_x1:tvl_x2]
            middle_mask_patch = cv.resize(middle_mask_patch, (volume_dimensions, volume_dimensions),
                                          interpolation=cv.INTER_NEAREST)
            middle_mask_patch[middle_mask_patch > 0] = foreground_pixel_value
            cv.imwrite(mask_output_path, middle_mask_patch)

            patch_info = {
                'filename': filename,
                'box': {
                    'y1': tvl_y1,
                    'x1': tvl_x1,
                    'y2': tvl_y2,
                    'x2': tvl_x2,
                }
            }
            patch_info_output_path = os.path.join(output_patch_information_folder, output_filename + '.txt')
            with open(patch_info_output_path, 'w') as f:
                json.dump(patch_info, f)

        __gt = gt_stabilized_masks[middle_frame_number]
        __mask = stabilized_detection_2d_masks[middle_frame_number]
        __frame = frames_list[middle_frame_number]

        __gt_output_path = os.path.join(gt_output_folder, f'{filename_wo_ext}.png')
        __mask_output_path = os.path.join(mask_2d_output_folder, f'{filename_wo_ext}.png')
        __frame_output_path = os.path.join(frame_output_folder, f'{filename_wo_ext}.png')

        cv.imwrite(__gt_output_path, __gt)
        cv.imwrite(__mask_output_path, __mask)
        cv.imwrite(__frame_output_path, __frame)


def main():
    pool = multiprocessing.Pool(os.cpu_count() - 1)

    output_volumes_folder = os.path.join(output_folder, 'volumes')
    output_masks_folder = os.path.join(output_folder, 'masks')
    output_patch_information_folder = os.path.join(output_folder, 'patch_information')

    os.makedirs(gt_output_folder, exist_ok=True)
    os.makedirs(mask_2d_output_folder, exist_ok=True)
    os.makedirs(frame_output_folder, exist_ok=True)

    os.makedirs(output_volumes_folder, exist_ok=True)
    os.makedirs(output_masks_folder, exist_ok=True)
    os.makedirs(output_patch_information_folder, exist_ok=True)

    frame_paths = glob(os.path.join(frames_folder, '*'))
    frame_paths.sort(reverse=True)
    video_wise_frame_paths = get_video_wise_list(frame_paths)
    for video_name in tqdm(video_wise_frame_paths.keys(), desc='Processing videos'):
        video_frames = video_wise_frame_paths[video_name]
        video_frames.sort()

        frame_numbers = list(range(0, len(video_frames) - frames_in_volume - 1, 1))

        params = []
        for frame_number in frame_numbers:
            params.append({
                'video_frames': video_frames,
                'frame_number': frame_number,
                'output_volumes_folder': output_volumes_folder,
                'output_masks_folder': output_masks_folder,
                'output_patch_information_folder': output_patch_information_folder,
            })

        for _ in tqdm(pool.imap_unordered(main_process, params), 'Processing frames', len(params)):
            pass


if __name__ == '__main__':
    main()
