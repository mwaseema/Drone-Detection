import json
import math
import multiprocessing
import os
from pathlib import Path
from shutil import copyfile

import cv2
from tqdm import tqdm


def get_files_from_folder(folder_path):
    all_files = []

    for fl in Path(folder_path).iterdir():
        filename = fl.name
        file_path = os.path.join(folder_path, filename)
        all_files.append(file_path)

    return all_files


def write_dict_to_json_file(file_path, dictionary_obj):
    with open(file_path, 'w') as fp:
        json.dump(dictionary_obj, fp)


def get_frames_and_ground_truths_from_folder(frames_folder, ground_truths_folder):
    all_frame_paths = get_files_from_folder(frames_folder)

    frame_ground_truth_paths = []
    for frame_path in all_frame_paths:
        frame_filename = os.path.basename(frame_path)
        ground_truth_path = os.path.join(ground_truths_folder, frame_filename)

        frame_ground_truth_paths.append({
            'frame': frame_path,
            'ground_truth': ground_truth_path
        })

    return frame_ground_truth_paths


def get_frame_and_ground_truth_crop_4splits(frame_path, ground_truth_path=None):
    extra_area = 50

    frame = cv2.imread(frame_path)
    ground_truth = cv2.imread(ground_truth_path)

    height, width, _ = frame.shape
    mid_height = math.floor(height / 2)
    mid_width = math.floor(width / 2)

    # top left
    frame1 = frame[0:mid_height + extra_area, 0:mid_width + extra_area, :]

    # top right
    frame2 = frame[0:mid_height + extra_area, mid_width - extra_area:width, :]

    # bottom left
    frame3 = frame[mid_height - extra_area:height, 0:mid_width + extra_area, :]

    # bottom right
    frame4 = frame[mid_height - extra_area:height, mid_width - extra_area:width, :]

    if ground_truth_path is not None:
        # top left
        ground_truth1 = ground_truth[0:mid_height + extra_area, 0:mid_width + extra_area, :]
        # top right
        ground_truth2 = ground_truth[0:mid_height + extra_area, mid_width - extra_area:width, :]
        # bottom left
        ground_truth3 = ground_truth[mid_height - extra_area:height, 0:mid_width + extra_area, :]
        # bottom right
        ground_truth4 = ground_truth[mid_height - extra_area:height, mid_width - extra_area:width, :]

        return [frame1, frame2, frame3, frame4], [ground_truth1, ground_truth2, ground_truth3, ground_truth4]
    else:
        return [frame1, frame2, frame3, frame4]


def get_frame_and_ground_truth_crop(frame_path, ground_truth_path=None):
    extra_area = 50

    frame = cv2.imread(frame_path)
    ground_truth = cv2.imread(ground_truth_path)

    height, width, _ = frame.shape
    split_height = math.floor(height / 3)
    split_width = math.floor(width / 3)

    # top left
    frame1 = frame[0:split_height + extra_area, 0:split_width + extra_area, :]
    # top middle
    frame2 = frame[0:split_height + extra_area, split_width - extra_area:split_width + split_width + extra_area, :]
    # top right
    frame3 = frame[0:split_height + extra_area, (split_width + split_width) - extra_area:width, :]
    # middle left
    frame4 = frame[split_height - extra_area:split_height + split_height + extra_area, 0:split_width + extra_area, :]
    # middle middle
    frame5 = frame[split_height - extra_area:split_height + split_height + extra_area,
             split_width - extra_area:split_width + split_width + extra_area, :]
    # middle right
    frame6 = frame[split_height - extra_area:split_height + split_height + extra_area,
             (split_width + split_width) - extra_area:width, :]
    # bottom left
    frame7 = frame[(split_height + split_height) - extra_area:height, 0:split_width + extra_area, :]
    # bottom middle
    frame8 = frame[(split_height + split_height) - extra_area:height,
             split_width - extra_area:split_width + split_width + extra_area, :]
    # bottom right
    frame9 = frame[(split_height + split_height) - extra_area:height, (split_width + split_width) - extra_area:width, :]

    if ground_truth_path is not None:
        # top left
        ground_truth1 = ground_truth[0:split_height + extra_area, 0:split_width + extra_area, :]
        # top middle
        ground_truth2 = ground_truth[0:split_height + extra_area,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # top right
        ground_truth3 = ground_truth[0:split_height + extra_area, (split_width + split_width) - extra_area:width, :]
        # middle left
        ground_truth4 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        0:split_width + extra_area, :]
        # middle middle
        ground_truth5 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # middle right
        ground_truth6 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        (split_width + split_width) - extra_area:width, :]
        # bottom left
        ground_truth7 = ground_truth[(split_height + split_height) - extra_area:height, 0:split_width + extra_area, :]
        # bottom middle
        ground_truth8 = ground_truth[(split_height + split_height) - extra_area:height,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # bottom right
        ground_truth9 = ground_truth[(split_height + split_height) - extra_area:height,
                        (split_width + split_width) - extra_area:width, :]

        return [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9], [ground_truth1, ground_truth2,
                                                                                          ground_truth3, ground_truth4,
                                                                                          ground_truth5, ground_truth6,
                                                                                          ground_truth7, ground_truth8,
                                                                                          ground_truth9]
    else:
        return [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9]


def crop_and_save_cropped_data(input_frames_folder, input_ground_truth_folder, output_frames_folder,
                               output_ground_truth_folder):
    os.makedirs(output_frames_folder, exist_ok=True)
    os.makedirs(output_ground_truth_folder, exist_ok=True)

    # get all the frames and ground truths
    frames_and_ground_truth_paths = get_frames_and_ground_truths_from_folder(input_frames_folder,
                                                                             input_ground_truth_folder)

    pbar = tqdm(total=len(frames_and_ground_truth_paths), desc="Croping and saving", unit=" frame", dynamic_ncols=True)
    for frame_ground_truth_path in frames_and_ground_truth_paths:
        frame_crops, ground_truth_crops = get_frame_and_ground_truth_crop(frame_ground_truth_path['frame'],
                                                                          frame_ground_truth_path['ground_truth'])
        frame_filename = os.path.basename(frame_ground_truth_path['frame'])
        frame_filename_without_ext, frame_ext = os.path.splitext(frame_filename)

        # copy original frame and ground truth
        copyfile(frame_ground_truth_path['frame'], os.path.join(output_frames_folder, frame_filename))
        copyfile(frame_ground_truth_path['ground_truth'], os.path.join(output_ground_truth_folder, frame_filename))

        # for copying cropped files
        for i in range(len(frame_crops)):
            crop_filename = f"{frame_filename_without_ext}_crop_{i + 1}{frame_ext}"
            # saving cropped frame
            cv2.imwrite(os.path.join(output_frames_folder, crop_filename), frame_crops[i])
            # saving ground truth frame
            cv2.imwrite(os.path.join(output_ground_truth_folder, crop_filename), ground_truth_crops[i])

        pbar.update()
    pbar.close()


def crop_and_save_cropped_data_multiprocessing_single(params):
    frame_ground_truth_path, output_frames_folder, output_ground_truth_folder = params

    frame_crops, ground_truth_crops = get_frame_and_ground_truth_crop(frame_ground_truth_path['frame'],
                                                                      frame_ground_truth_path['ground_truth'])
    frame_filename = os.path.basename(frame_ground_truth_path['frame'])
    frame_filename_without_ext, frame_ext = os.path.splitext(frame_filename)

    # copy original frame and ground truth
    copyfile(frame_ground_truth_path['frame'], os.path.join(output_frames_folder, frame_filename))
    copyfile(frame_ground_truth_path['ground_truth'], os.path.join(output_ground_truth_folder, frame_filename))

    # for copying cropped files
    for i in range(len(frame_crops)):
        crop_filename = f"{frame_filename_without_ext}_crop_{i + 1}{frame_ext}"
        # saving cropped frame
        cv2.imwrite(os.path.join(output_frames_folder, crop_filename), frame_crops[i])
        # saving ground truth frame
        cv2.imwrite(os.path.join(output_ground_truth_folder, crop_filename), ground_truth_crops[i])


def crop_and_save_cropped_data_multiprocessing(input_frames_folder: str, input_ground_truth_folder: str,
                                               output_frames_folder: str, output_ground_truth_folder: str):
    os.makedirs(output_frames_folder, exist_ok=True)
    os.makedirs(output_ground_truth_folder, exist_ok=True)

    # get all the frames and ground truths
    frames_and_ground_truth_paths = get_frames_and_ground_truths_from_folder(input_frames_folder,
                                                                             input_ground_truth_folder)

    # making params for multiprocess functioning
    params = []
    for i in range(len(frames_and_ground_truth_paths)):
        t = (frames_and_ground_truth_paths[i], output_frames_folder, output_ground_truth_folder)
        params.append(t)

    pool = multiprocessing.Pool(os.cpu_count() - 1)
    for _ in tqdm(pool.imap_unordered(crop_and_save_cropped_data_multiprocessing_single, params), total=len(params),
                  desc="Generating dataset crops", unit='image'):
        pass
    pool.close()
    pool.join()
