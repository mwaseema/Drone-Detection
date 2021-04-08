import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

videos_folder = ''
optical_flow_output_folder = ''
optical_flow_for_every_nth_frame = 4


def generate_optical_flow(first_frame: np.ndarray, second_frame: np.ndarray):
    optical_flow = cv2.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(first_frame, second_frame, None)

    f1_f = flow[..., 0]
    f2_f = flow[..., 1]

    f1_f_g = np.gradient(np.array(f1_f, dtype=float))
    f2_f_g = np.gradient(np.array(f2_f, dtype=float))

    Ux_f = f1_f_g[0]
    Uy_f = f1_f_g[1]
    Vx_f = f2_f_g[0]
    Vy_f = f2_f_g[1]

    U_f_mag = np.square(Ux_f) + np.square(Uy_f)
    V_f_mag = np.square(Vx_f) + np.square(Vy_f)

    U_f_mag = np.sqrt(U_f_mag)
    V_f_mag = np.sqrt(V_f_mag)

    MB = np.maximum(U_f_mag, V_f_mag)

    MB_norm = 255 * cv2.normalize(MB, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return MB_norm


def main():
    video_file_paths = glob(os.path.join(videos_folder, '*.mov'))

    for video_file_path in tqdm(video_file_paths, desc="Processing video", unit='Video', dynamic_ncols=True):
        video_filename = os.path.basename(video_file_path)
        video_filename_without_ext, video_ext = os.path.splitext(video_filename)

        frame_cnt = 0
        previous_frame = None
        video_cap = cv2.VideoCapture(video_file_path)
        pbar = tqdm(total=video_cap.get(cv2.CAP_PROP_FRAME_COUNT),
                    desc=f'Processing Frame of video {video_filename_without_ext}', unit='Frame',
                    dynamic_ncols=True)
        while True:
            grabbed, frame = video_cap.read()

            if not grabbed:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # if frame count it divisible by n and previous frame also exists
            if frame_cnt % optical_flow_for_every_nth_frame == 0 and previous_frame is not None:
                optical_flow = generate_optical_flow(previous_frame, frame)

                output_flow_file_path = os.path.join(optical_flow_output_folder, video_filename_without_ext,
                                                     f"{video_filename_without_ext}_{frame_cnt:0>6}.png")
                os.makedirs(os.path.dirname(output_flow_file_path), exist_ok=True)
                cv2.imwrite(output_flow_file_path, optical_flow)

            previous_frame = frame
            frame_cnt += 1

            pbar.update()
        pbar.close()

        video_cap.release()


if __name__ == "__main__":
    main()
