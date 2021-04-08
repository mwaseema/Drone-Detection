import os
from glob import glob

import numpy as np
import cv2
from tqdm import tqdm

SMOOTHING_RADIUS = 50

OUTPUT_BASE = ''

VIDEO_FILES_FOLDER = ''
VIDEO_FILES_OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, 'videos')

STABALIZED_FRAMES_OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, 'frames')

# not written if GROUND_TRUTH_FOLDER is empty
GROUND_TRUTH_FOLDER = ''
GROUND_TRUTH_OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, 'ground_truth')

# not written if DETECTION_MASKS_FOLDER is empty
DETECTION_MASKS_FOLDER = ''
DETECTION_MASKS_OUTPUT_FOLDER = os.path.join(OUTPUT_BASE, 'detection_masks')


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def stabilize_video_motion(video_file_path, video_output_path, frames_output_folder, ground_truth_folder,
                           ground_truth_output_folder, detection_folder, detection_output_folder):
    video_filename = os.path.basename(video_file_path)
    video_filename_wo_ext = os.path.splitext(video_filename)[0]

    # Read input video
    cp = cv2.VideoCapture(video_file_path)

    # To get number of frames
    n_frames = int(cp.get(cv2.CAP_PROP_FRAME_COUNT))

    # To check the number of frames in the video
    print("Number of frames", n_frames)

    width = int(cp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cp.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("width", width)
    print("height", height)

    # get the number of frames per second
    fps = cp.get(cv2.CAP_PROP_FPS)

    # Try doing 2*width
    out = cv2.VideoWriter(filename=video_output_path, fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=float(fps),
                          frameSize=(width, height), isColor=True)

    # read the first frame
    _, prev = cp.read()

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in tqdm(list(range(n_frames - 2)), desc="Finding transforms"):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=30, blockSize=3)

        succ, curr = cp.read()

        if not succ:
            break

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

        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]

        prev_gray = curr_gray

    # print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

    # Find the cumulative sum of tranform matrix for each dx,dy and da
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    cp.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Write n_frames-1 transformed frames
    for i in tqdm(list(range(n_frames - 2)), desc='Writing transformed frames'):
        frame_filename = f"{video_filename_wo_ext}_{str(i).zfill(6)}.png"
        frames_output_path = os.path.join(frames_output_folder, frame_filename)

        # Read next frame
        success, frame = cp.read()
        if not success:
            break

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

        # How to add inverse transformation matrix if required
        # m_inv = np.zeros((2, 3), np.float32)
        # m_inv[0, 0] = np.cos(da)
        # m_inv[0, 1] = np.sin(da)
        # m_inv[1, 0] = -np.sin(da)
        # m_inv[1, 1] = np.cos(da)
        # m_inv[0, 2] = -dx * np.cos(da) - dy * np.sin(da)
        # m_inv[1, 2] = -dy * np.cos(da) + dx * np.sin(da)
        # frame_inv = frame_stabilized.copy()
        # frame_inv = cv2.warpAffine(frame_inv, m_inv, (width, height))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # Write the frame to the file
        # frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
        # if (frame_out.shape[1] > 1920):
        #     frame_out = cv2.resize(frame_out, (frame_out.shape[1] / 2, frame_out.shape[0] / 2))

        # cv2.imshow("Before and After", frame_out)
        # cv2.waitKey(10)

        # out.write(frame_out)
        out.write(frame_stabilized)

        # cv2.imwrite(frames_output_path, frame_stabilized)

        if ground_truth_folder != '':
            ground_truth_path = os.path.join(ground_truth_folder, frame_filename)
            ground_truth_output_path = os.path.join(ground_truth_output_folder, frame_filename)
            if os.path.exists(ground_truth_path):
                gt = cv2.imread(ground_truth_path)
                gt_stabilized = cv2.warpAffine(gt, m, (width, height))
                gt_stabilized = fixBorder(gt_stabilized)
                cv2.imwrite(ground_truth_output_path, gt_stabilized)

        if detection_folder != '':
            detection_path = os.path.join(detection_folder, frame_filename)
            detection_output_path = os.path.join(detection_output_folder, frame_filename)
            if os.path.exists(detection_path):
                det = cv2.imread(detection_path)
                det_stabilized = cv2.warpAffine(det, m, (width, height))
                det_stabilized = fixBorder(det_stabilized)
                cv2.imwrite(detection_output_path, det_stabilized)

    # Release video
    cp.release()
    out.release()
    # Close windows
    cv2.destroyAllWindows()


def main():
    os.makedirs(VIDEO_FILES_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(STABALIZED_FRAMES_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(GROUND_TRUTH_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DETECTION_MASKS_OUTPUT_FOLDER, exist_ok=True)

    video_paths = glob(os.path.join(VIDEO_FILES_FOLDER, '*'))
    for video_path in tqdm(video_paths, desc='Processing videos', unit='video'):
        filename = os.path.basename(video_path)
        filename_wo_ext = os.path.splitext(filename)[0]

        output_file_path = os.path.join(VIDEO_FILES_OUTPUT_FOLDER, f'{filename_wo_ext}.avi')
        stabilize_video_motion(video_path, output_file_path, STABALIZED_FRAMES_OUTPUT_FOLDER, GROUND_TRUTH_FOLDER,
                               GROUND_TRUTH_OUTPUT_FOLDER, DETECTION_MASKS_FOLDER, DETECTION_MASKS_OUTPUT_FOLDER)


if __name__ == '__main__':
    main()
