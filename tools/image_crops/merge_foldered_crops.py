import os
from glob import glob
from typing import List

import cv2
from tqdm import tqdm

from tools.image_crops.merge_funcs import merge_segmentation_masks


class MergeFolderedCrops:
    def __init__(self):
        self.folder_containing_foldered_crops = ''
        self.output_folder = ''

        self.number_of_crops = 9

    def get_available_filenames(self) -> List[str]:
        """
        Get all the filenames from one folder so these can be used to get files from other folders
        :return:
        """
        # getting file paths from a folder
        file_paths = glob(os.path.join(self.folder_containing_foldered_crops, '0', '*'))
        filenames = []
        for fp in file_paths:
            filename = os.path.basename(fp)
            filenames.append(filename)
        return filenames

    def get_paths_to_crops_of_single_mask(self, filenames: List[str]) -> List[List[str]]:
        """
        Generate list with corresponding paths to all the crops for single map
        :param filenames: Names of the files available in any of the folder
        :return: List containing paths to all the crops as list
        """

        corresponding_file_crops = []

        for filename in filenames:
            crops = []
            for i in range(self.number_of_crops):
                crop_path = os.path.join(self.folder_containing_foldered_crops, str(i), filename)
                crops.append(crop_path)
            corresponding_file_crops.append(crops)
        return corresponding_file_crops

    def main(self):
        os.makedirs(self.output_folder, exist_ok=True)

        # get all file names
        filenames = self.get_available_filenames()
        paths_with_corresponding_crops = self.get_paths_to_crops_of_single_mask(filenames)

        for pwcc in tqdm(paths_with_corresponding_crops):
            merge_filename = None
            crops_to_merge = []
            for crop_path in pwcc:
                if merge_filename is None:
                    merge_filename = os.path.basename(crop_path)
                crops_to_merge.append(cv2.imread(crop_path))

            merged = merge_segmentation_masks(crops_to_merge)
            output_path = os.path.join(self.output_folder, merge_filename)
            cv2.imwrite(output_path, merged)


if __name__ == '__main__':
    r = MergeFolderedCrops()
    r.main()
