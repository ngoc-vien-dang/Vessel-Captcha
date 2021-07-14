"""
File name: grid.py
Author: ngocviendang
Date created: July 13, 2020

This file creat the grid for the images.
"""
import argparse
import sys
import os
import numpy as np
import nibabel as nib
from captcha.utils.helper import load_nifti_mat_from_file
from captcha.utils.helper import create_and_save_nifti
from captcha.utils.helper import getAllFiles

def main(args):
    original_data_dir = args.original_data
    grid_filepath = args.grid_filepath
    patch_size = 32
    if not os.path.exists(grid_filepath):
        os.makedirs(grid_filepath)
    unfiltered_filelist = getAllFiles(original_data_dir)
    input_list = [item for item in unfiltered_filelist if re.search('_img', item)]
    mask_list = [item for item in unfiltered_filelist if re.search('_mask', item)]
    input_list = sorted(input_list)
    mask_list = sorted(mask_list)
    print(input_list)
    print(mask_list)
    # load image, mask and label stacks as matrices
    for i, j in enumerate(input_list):
        print('> Loading image...')
        img_mat = load_nifti_mat_from_file(j)
        print('Loading mask...')
        mask_mat = load_nifti_mat_from_file(mask_list[i])
        # the grid is going to be saved in this matrix
        prob_mat = np.zeros(img_mat.shape, dtype=np.float32)
        x_dim, y_dim, z_dim = prob_mat.shape
        # get the x, y and z coordinates where there is brain
        x, y, z = np.where(mask_mat)
        print('x shape:', x.shape)
        print('y shape:', y.shape)
        print('z shape:', z.shape)
        # get the z slices with brain
        z_slices = np.unique(z)
        # proceed slice by slice
        for l in z_slices:
            slice_vox_inds = np.where(z == l)
            # find all x and y coordinates with brain in given slice
            x_in_slice = x[slice_vox_inds]
            y_in_slice = y[slice_vox_inds]
            # find min and max x and y coordinates
            slice_x_min = min(x_in_slice)
            slice_x_max = max(x_in_slice)
            slice_y_min = min(y_in_slice)
            slice_y_max = max(y_in_slice)
            # calculate number of patches in x and y direction in given slice
            num_of_x_patches = np.int(np.ceil((slice_x_max - slice_x_min) / patch_size))
            num_of_y_patches = np.int(np.ceil((slice_y_max - slice_y_min) / patch_size))
            for m in range(num_of_x_patches):
                for n in range(num_of_y_patches):
                    # find the starting and ending x and y coordinates of given patch
                    patch_start_x = slice_x_min + patch_size * m
                    patch_end_x = slice_x_min + patch_size * (m + 1)
                    patch_start_y = slice_y_min + patch_size * n
                    patch_end_y = slice_y_min + patch_size * (n + 1)
                    if patch_end_x > x_dim:
                        patch_end_x = slice_x_max
                        patch_start_x = slice_x_max - patch_size
                    if patch_end_y > y_dim:
                        patch_end_y = slice_y_max
                        patch_start_y = slice_y_max - patch_size
                    prob_mat[patch_start_x: patch_end_x, patch_start_y, l] = 1
                    prob_mat[patch_start_x: patch_end_x, patch_end_y, l] = 1
                    prob_mat[patch_start_x, patch_start_y: patch_end_y, l] = 1
                    prob_mat[patch_end_x, patch_start_y: patch_end_y, l] = 1
        # SAVE AS NIFTI
        create_and_save_nifti(prob_mat, grid_filepath + j.split(os.sep)[-1].split('_')[0] + '_grid.nii')
    print('DONE')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data", type=str,
                        help='the filename path of the training set.')
    parser.add_argument("--grid_filepath", type=str,
                        help='The filename path for saving the grid-mask matrix.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
