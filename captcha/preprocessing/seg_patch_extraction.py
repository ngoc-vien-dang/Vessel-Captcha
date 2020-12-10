"""
File name: seg_patch_extraction.py
Author: ngocviendang
Date created: July 13, 2020
This file extracts the patches of brain volumes.
Reference https://github.com/prediction2020/unet-vessel-segmentation
"""
import os
import numpy as np
import nibabel as nib
import sys
import argparse
from captcha.utils.helper import load_nifti_mat_from_file
from captcha.utils.helper import gen_filename_pairs_3

def main(args):
    patch_sizes = [96]    # different quadratic patch sizes n x n
    nr_patches = 1000  # number of patches we want to extract from one stack (one patient)
    nr_vessel_patches = nr_patches // 2  # patches that are extracted around vessels
    nr_empty_patches = nr_patches - nr_vessel_patches  # patches that are extracted from the brain region but not around
    # vessels
    # extract patches from each data stack (patient)
    skull_stripping_dir = os.path.expanduser(args.skull_stripping_dir)
    patch_extraction_dir = os.path.expanduser(args.patch_extraction_dir)
    if not os.path.exists(patch_extraction_dir):
        os.makedirs(patch_extraction_dir)
    # List filenames of data after the skull stripping process
    input_list, mask_list, label_list = gen_filename_pairs_3(skull_stripping_dir, 'img', 'mask', 'label')
    input_list = sorted(input_list)
    mask_list = sorted(mask_list)
    label_list = sorted(label_list)
    print(input_list)
    print(mask_list)
    print(label_list)
    # load image, mask and label stacks as matrices	
    for i,j in enumerate(input_list):
        # load image and label stacks as matrices
        print('> Loading image...')
        img_mat = load_nifti_mat_from_file(j)
        print('> Loading mask...')
        mask_mat = load_nifti_mat_from_file(mask_list[i])
        print('> Loading label...')
        label_mat = load_nifti_mat_from_file(label_list[i])  # values 0 or 1

        current_nr_extracted_patches = 0  # counts already extracted patches
        img_patches = {}  # dictionary to save image patches
        label_patches = {}  # dictionary to save label patches
        # make lists in dictionaries for each extracted patch size
        for size in patch_sizes:
            img_patches[str(size)] = []
            label_patches[str(size)] = []

        # variables with sizes and ranges for searchable areas
        max_patch_size = max(patch_sizes)
        print('max_patch_size: ', max_patch_size)
        half_max_size = max_patch_size // 2
        print('half_max_size: ', half_max_size)
        max_row = label_mat.shape[0] - max_patch_size // 2
        print('max_row: ', max_row)
        max_col = label_mat.shape[1] - max_patch_size // 2
        print('max_col: ', max_col)
	  
        # -----------------------------------------------------------
        # EXTRACT RANDOM PATCHES WITH VESSELS IN THE CENTER OF EACH PATCH
        # -----------------------------------------------------------
        # cut off half of the biggest patch on the edges to create the searchable area -> to ensure that there will be
        # enough space for getting the patch
        searchable_label_area = label_mat[half_max_size: max_row, half_max_size: max_col, :]
        # find all vessel voxel indices in searchable area
        vessel_inds = np.asarray(np.where(searchable_label_area == 1))

        # keep extracting patches while the desired number of patches has not been reached yet, this just in case some
        # patches would be skipped, because they were already extracted (we do not want to have same patches in the set
        # more than once)
        while current_nr_extracted_patches < nr_vessel_patches * len(patch_sizes):
            # find given number of random vessel indices
            random_vessel_inds = vessel_inds[:,
                                 np.random.choice(vessel_inds.shape[1], nr_vessel_patches, replace=False)]
            for i in range(nr_vessel_patches):
                # stop extracting if the desired number of patches has been reached
                if current_nr_extracted_patches == nr_vessel_patches * len(patch_sizes):
                    break

                # get the coordinates of the random vessel around which the patch will be extracted
                x = random_vessel_inds[0][i] + half_max_size
                y = random_vessel_inds[1][i] + half_max_size
                z = random_vessel_inds[2][i]

                # extract patches of different quadratic sizes with the random vessel voxel in the center of each patch
                for size in patch_sizes:
                    half_size = size // 2
                    random_img_patch = img_mat[x - half_size:x + half_size, y - half_size:y + half_size, z]
                    random_label_patch = label_mat[x - half_size:x + half_size, y - half_size:y + half_size, z]

                    # just sanity check if the patch is already in the list
                    if any((random_img_patch == x).all() for x in img_patches[str(size)]):
                        print('Skip patch because already extracted. size:', size)
                        break
                    else:
                        # append the extracted patches to the dictionaries
                        img_patches[str(size)].append(random_img_patch)
                        label_patches[str(size)].append(random_label_patch)
                        current_nr_extracted_patches += 1
                        if current_nr_extracted_patches % 100 == 0:
                            print(current_nr_extracted_patches, 'PATCHES CREATED')

        # -----------------------------------------------------------
        # EXTRACT RANDOM EMPTY PATCHES
        # -----------------------------------------------------------
        # cut off half of the biggest patch on the edges to create the searchable area -> to ensure that there will be
        # enough space for getting the patch
        searchable_mask_area = mask_mat[half_max_size: max_row, half_max_size: max_col, :]
        # find all brain voxel indices
        brain_inds = np.asarray(np.where(searchable_mask_area == 1))

        # keep extracting patches while the desired number of patches has not been reached yet, this just in case some
        # patches would be skipped, because they were already extracted (we do not want to have same patches in the set
        # more than once)
        while current_nr_extracted_patches < nr_patches * len(patch_sizes):
            # find given number of random indices in the brain area
            random_brain_inds = brain_inds[:, np.random.choice(brain_inds.shape[1], nr_empty_patches, replace=False)]
            for i in range(nr_empty_patches):
                # stop extracting if the desired number of patches has been reached
                if current_nr_extracted_patches == nr_patches * len(patch_sizes):
                    break

                # get the coordinates of the random brain voxel around which the patch will be extracted
                x = random_brain_inds[0][i] + half_max_size
                y = random_brain_inds[1][i] + half_max_size
                z = random_brain_inds[2][i]

                # extract patches of different quadratic sizes with the random brain voxel in the center of each patch
                for size in patch_sizes:
                    half_size = size // 2
                    random_img_patch = img_mat[x - half_size:x + half_size, y - half_size:y + half_size, z]
                    random_label_patch = label_mat[x - half_size:x + half_size, y - half_size:y + half_size, z]

                    # just sanity check if the patch is already in the list
                    if any((random_img_patch == x).all() for x in img_patches[str(size)]):
                        print('Skip patch because already extracted. size:', size)
                        break
                    else:
                        # append the extracted patches to the dictionaries
                        img_patches[str(size)].append(random_img_patch)
                        label_patches[str(size)].append(random_label_patch)
                        current_nr_extracted_patches += 1
                        if current_nr_extracted_patches % 100 == 0:
                            print(current_nr_extracted_patches, 'PATCHES CREATED')

        assert current_nr_extracted_patches == nr_patches * len(
            patch_sizes), "The number of extracted patches is  " + str(
            current_nr_extracted_patches) + " but should be " + str(
            nr_patches * len(patch_sizes))

        # save extracted patches as numpy arrays
        for size in patch_sizes:
            print('number of extracted image patches:', len(img_patches[str(size)]))
            print('number of extracted label patches:', len(label_patches[str(size)]))
            directory = patch_extraction_dir
            np.save(directory + j.split(os.sep)[-1].split('_')[0] + '_' + str(size) + '_img', np.asarray(img_patches[str(size)]))
            np.save(directory + j.split(os.sep)[-1].split('_')[0] + '_' + str(size) + '_label', np.asarray(label_patches[str(size)]))
            print('Image patches saved to', directory + j.split(os.sep)[-1].split('_')[0] + '_' + str(size) + '_img.npy')
            print('Label patches saved to', directory + j.split(os.sep)[-1].split('_')[0] + '_' + str(size) + '_label.npy')
        print()
        
    print('DONE')
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--skull_stripping_dir", type=str, 
                        help='Directory for saving the images after the skull stripping process.')
    parser.add_argument("--patch_extraction_dir", type=str, 
                        help='Directory for saving the images after the patch extraction process.')
    return parser.parse_args(argv)
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))  
