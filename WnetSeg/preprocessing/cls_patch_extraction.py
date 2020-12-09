"""
File name: cls_patch_extraction.py
Author: ngocviendang
Date created: July 13, 2020

This file contains helper functions for other scripts.
"""
import argparse
import sys
import os
import random
import numpy as np
import nibabel as nib
import re
from sklearn import preprocessing
from WnetSeg.utils.helper import gen_filename_pairs_3
from WnetSeg.utils.helper import load_nifti_mat_from_file

def main(args):
    patch_size = args.patch_size
    skull_stripping_dir = os.path.expanduser(args.skull_stripping_dir)
    patch_vessel_dir = os.path.expanduser(args.patch_vessel_dir)
    if not os.path.exists(patch_vessel_dir):
        os.makedirs(patch_vessel_dir)
    # List filenames of data after the skull stripping process
    input_list, mask_list, label_list = gen_filename_pairs_3(
        skull_stripping_dir, 'img', 'mask', 'rerun')
    input_list = sorted(input_list)
    mask_list = sorted(mask_list)
    label_list = sorted(label_list)
    print(input_list)
    print(mask_list)
    print(label_list)
    # load image, mask and label stacks as matrices
    for i, j in enumerate(input_list):
        print('Loading image...')
        img_mat = load_nifti_mat_from_file(j)
        print('Loading mask...')
        mask_mat = load_nifti_mat_from_file(mask_list[i])
        print('Loading weak label...')
        label_mat = load_nifti_mat_from_file(label_list[i])
        vessel_patches = []  # list to save vessel patches
        empty_patches = []  # list to save label patches
        #img_patches_empty = []  # list to save wo-vessel patches
        # extract square
        prob_mat = np.zeros(img_mat.shape, dtype=np.float32)
        x_dim, y_dim, _ = prob_mat.shape
        # get the x, y and z coordinates where there is brain
        x, y, z = np.where(mask_mat)
        z_slices = np.unique(z)
        cnt = 0
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
            num_of_x_patches = np.int(
                np.ceil((slice_x_max - slice_x_min) / patch_size))
            num_of_y_patches = np.int(
                np.ceil((slice_y_max - slice_y_min) / patch_size))
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
                    # get the patch with the found coordinates from the image matrix
                    img_patch = img_mat[patch_start_x: patch_end_x,
                                        patch_start_y: patch_end_y, l]
                    label_patch = label_mat[patch_start_x: patch_end_x,
                                            patch_start_y: patch_end_y, l]
                    if 2 in label_patch:                   
                        cnt += 1
                        vessel_patches.append(img_patch)                           
                    else:
                        empty_patches.append(img_patch)
        # save extracted patches as numpy arrays
        print('the number of vessel patch:', cnt)
        print('the number of extracted img patches:', len(vessel_patches))
        print('the number of extracted empty patches:', len(empty_patches))
        np.save(patch_vessel_dir + j.split(os.sep)
                [-1].split('_')[0] + '_' + str(patch_size) + '_vessel', np.asarray(vessel_patches))
        np.save(patch_vessel_dir + j.split(os.sep)[-1].split('_')[
                0] + '_' + str(patch_size) + '_empty', np.asarray(empty_patches))
        print('img patches saved to', patch_vessel_dir + j.split(os.sep)
              [-1].split('_')[0] + '_' + str(patch_size) + '_vessel.npy')
        print('label patches saved to', patch_vessel_dir + j.split(os.sep)
              [-1].split('_')[0] + '_' + str(patch_size) + '_empty.npy')
    print()
    print('DONE')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int,
                        help='the quadratic patch sizes.')
    #parser.add_argument("--n_patches", type=int,
    #help='the number of patches we want to extract from one stack')
    parser.add_argument("--skull_stripping_dir", type=str,
                        help='Directory for saving the images after the skull stripping process.')
    parser.add_argument("--patch_vessel_dir", type=str,
                        help='Directory for saving the images after the patch extraction process.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
