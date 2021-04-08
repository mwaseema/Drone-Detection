import sys
from os import path

if path.abspath(path.join(path.dirname(__file__), '..', '..')) not in sys.path:
    sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', '..')))

from tools.image_crops.funts import crop_and_save_cropped_data_multiprocessing


class GenerateCrops:
    def __init__(self):
        # folder containing images
        self.input_frames_folder = ''
        # folder containing ground truth masks
        self.input_ground_truth_folder = ''

        # folder where to output cropped images
        self.output_frames_folder = ''
        # folder where to output cropped masks
        self.output_ground_truth_folder = ''

    def main(self):
        crop_and_save_cropped_data_multiprocessing(self.input_frames_folder, self.input_ground_truth_folder,
                                                   self.output_frames_folder, self.output_ground_truth_folder)


if __name__ == '__main__':
    generate_crops = GenerateCrops()
    generate_crops.main()
