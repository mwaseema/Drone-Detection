from typing import List

import cv2
import numpy as np

from tools.motion_stablization.stabalize_motion import smooth


def get_motion_stabilization(frames: List[np.ndarray], masks: List[np.ndarray] = None,
                             gt_masks: List[np.ndarray] = None):
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

    return stabilized_frames, inverse_matrices, stabilized_masks, gt_stabilized_masks


def apply_inverse_transformations(frame: np.ndarray, inverse_matrix: np.ndarray, height, width):
    frame = frame.copy()
    frame_inv = cv2.warpAffine(frame, inverse_matrix, (width, height))
    return frame_inv
