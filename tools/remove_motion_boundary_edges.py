import os
from glob import glob

import cv2 as cv
from tqdm import tqdm

mb_folder = ''
output_folder = ''
remove_pixels = 10

os.makedirs(output_folder, exist_ok=True)

mb_paths = glob(mb_folder + '/*')
for mb_path in tqdm(mb_paths):
    filename = os.path.basename(mb_path)
    output_path = os.path.join(output_folder, filename)

    mb = cv.imread(mb_path)
    height, width = mb.shape[0:2]

    # remove from top
    mb[0:remove_pixels, :, :] = 0
    # remove from bottom
    mb[height - remove_pixels:height, :, :] = 0
    # remove from left
    mb[:, 0:remove_pixels, :] = 0
    # remove from right
    mb[:, width - remove_pixels:width, :] = 0

    cv.imwrite(output_path, mb)
