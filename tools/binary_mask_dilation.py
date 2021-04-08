import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

mb_folder = ''
output_folder = ''
mb_threshold = 30
dilation_value = 5

os.makedirs(output_folder, exist_ok=True)

mb_paths = glob(mb_folder + '/*')
for mb_path in tqdm(mb_paths):
    filename = os.path.basename(mb_path)

    mb = cv2.imread(mb_path)
    mb[mb < mb_threshold] = 0
    mb[mb >= mb_threshold] = 255

    mb = cv2.dilate(mb, np.ones((dilation_value, dilation_value), np.uint8))

    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, mb)
