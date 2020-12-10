"""
File name: prediction.py
Author: ngocviendang
Date created: July 21, 2020

This is the main script for classifying vessel patch and non-vessel patch to:
Patch-annotation automatically for MRA image
Noise filer for vessel segmentation 
"""
import pickle
import time
import numpy as np
from keras.models import load_model
import os
import argparse
import sys
import nibabel as nib
from captcha.utils.helper import create_and_save_nifti
from captcha.utils.helper import gen_filename_pairs_2
from captcha.utils.helper import load_nifti_mat_from_file

def main(args):
    train_non_label_dir = args.train_non_label_dir
    patch_size = args.patch_size
    model_filepath = args.model_filepath
    train_metadata_filepath = args.train_metadata_filepath
    grid_label_filepath = args.grid_label_filepath
    # LOADING MODEL, RESULTS AND WHOLE BRAIN MATRICES
    print(model_filepath)
    model = load_model(model_filepath)
    with open(train_metadata_filepath, 'rb') as handle:
        train_metadata = pickle.load(handle)
    print(train_metadata)
    # List filenames of data after the skull stripping process
    input_list, mask_list = gen_filename_pairs_2(train_non_label_dir, 'img', 'mask')
    input_list = sorted(input_list)
    mask_list = sorted(mask_list)
    print(input_list)
    print(mask_list)
    # load image, mask and label stacks as matrices
    for i, j in enumerate(input_list):
        print('Loading image...')
        img_mat = load_nifti_mat_from_file(j)
        print('Loading mask...')
        mask_mat = load_nifti_mat_from_file(mask_list[i])
        # weak_annotation matrix
        prob_mat = np.zeros(img_mat.shape, dtype=np.float32)
        x_dim, y_dim, _ = prob_mat.shape
        # get the x, y and z coordinates where there is brain
        x, y, z = np.where(mask_mat)
        print('x shape:', x.shape)
        print('y shape:', y.shape)
        print('z shape:', z.shape)
        # get the z slices with brain
        z_slices = np.unique(z)
        # start cutting out and predicting the patches
        # proceed slice by slice
        for l in z_slices:
            print('Slice:', l)
            slice_vox_inds = np.where(z == l)
            # find all x and y coordinates with brain in given slice
            x_in_slice = x[slice_vox_inds]
            y_in_slice = y[slice_vox_inds]
            # find min and max x and y coordinates
            slice_x_min = min(x_in_slice)
            slice_x_max = max(x_in_slice)
            slice_y_min = min(y_in_slice)
            slice_y_max = max(y_in_slice)
            # calculate number of predicted patches in x and y direction in given slice
            num_of_x_patches = np.int(np.ceil((slice_x_max - slice_x_min) / patch_size))
            num_of_y_patches = np.int(np.ceil((slice_y_max - slice_y_min) / patch_size))
            print('num x patches', num_of_x_patches)
            print('num y patches', num_of_y_patches)
            # predict patch by patch in given slice
            for m in range(num_of_x_patches):
                for n in range(num_of_y_patches):
                    # find the starting and ending x and y coordinates of given patch
                    patch_start_x = slice_x_min + patch_size * m
                    patch_end_x = slice_x_min + patch_size * (m + 1)
                    patch_start_y = slice_y_min + patch_size * n
                    patch_end_y = slice_y_min + patch_size * (n + 1)
                    # if the dimensions of the probability matrix are exceeded shift back the last patch
                    if patch_end_x > x_dim:
                        patch_end_x = slice_x_max
                        patch_start_x = slice_x_max - patch_size
                    if patch_end_y > y_dim:
                        patch_end_y = slice_y_max
                        patch_start_y = slice_y_max - patch_size
                    # get the patch with the found coordinates from the image matrix
                    img_patch = img_mat[patch_start_x: patch_end_x,
                                        patch_start_y: patch_end_y, l]
                    # normalize the patch with mean and standard deviation calculated over training set
                    img_patch = img_patch.astype(np.float)
                    img_patch = img_patch[None,:,:,None]
                    img_patch -= train_metadata['mean_train']
                    img_patch /= train_metadata['std_train']
                    if model.predict(img_patch) >= 0.5:
                        prob_mat[patch_start_x: patch_end_x,patch_start_y: patch_end_y, l] = 2
                    prob_mat[patch_start_x: patch_end_x, patch_start_y, l] = 1
                    prob_mat[patch_start_x: patch_end_x, patch_end_y, l] = 1
                    prob_mat[patch_start_x, patch_start_y: patch_end_y, l] = 1
                    prob_mat[patch_end_x, patch_start_y: patch_end_y, l] = 1
        # Save weak annotation
        create_and_save_nifti(prob_mat, grid_label_filepath +
                              j.split(os.sep)[-1].split('_')[0]+'_rerun.nii')
        print('done')
    print('DONE')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_non_label_dir", type=str,
                        help='the filename path of the training set having no labels or the test set.')
    parser.add_argument("--patch_size", type=int,
                        help='the quadratic patch sizes.')
    parser.add_argument("--train_metadata_filepath", type=str,
                        help='The filename path for loading the mean and std of training set of pnetcls model.')
    parser.add_argument("--model_filepath", type=str,
                        help='The filename path for loading the pnetcls model.')
    parser.add_argument("--grid_label_filepath", type=str,
                        help='The filename path for saving the patch-mask matrix.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    

