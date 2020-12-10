"""
File name: rough_mask.py
Author: ngocviendang
Date created: July 13, 2020

This file defines the method to extract the rough mask of brain volumes in training set.
"""
import argparse
import sys
import os
import numpy as np
import nibabel as nib
import random
import re
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture as GMM
from captcha.utils.helper import gen_filename_pairs_3
from captcha.utils.helper import load_nifti_mat_from_file
from captcha.utils.helper import create_and_save_nifti
import cv2

def main(args):
    patch_size = args.patch_size
    clustering = args.clustering
    patch_annotation_dir = os.path.expanduser(args.patch_annotation_dir)
    rough_mask_dir = os.path.expanduser(args.rough_mask_dir)
    if not os.path.exists(rough_mask_dir):
        os.makedirs(rough_mask_dir)
    # List filenames of data after the skull stripping process
    input_list, mask_list, label_list = gen_filename_pairs_3(patch_annotation_dir, 'img', 'mask', 'rerun')
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
        # Normalization
        mask = img_mat > 0
        img_mat = (img_mat - img_mat[mask].mean()) / img_mat[mask].std()
        print('Loading mask...')
        mask_mat = load_nifti_mat_from_file(mask_list[i])
        print('Loading weak label...')
        label_mat = load_nifti_mat_from_file(label_list[i])
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
                    mask_check = mask_mat[patch_start_x: patch_end_x,
                                          patch_start_y: patch_end_y, l]
                    if 2 in label_patch:
                        image = img_patch
                        pixels = image
                        pixels = pixels.astype('float32')
                        if clustering == 'kmeans':
                            try:
                                vectorized = pixels.reshape((-1, 1))
                                vectorized = np.float32(vectorized)
                                criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                                attempts = 10
                                if 0 in mask_check:
                                    K = 4
                                else:
                                    K = 2
                                _, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
                                res = center[label.flatten()]
                                result_image = res.reshape((image.shape))
                                result_image[result_image != np.amax(center)] = 0
                                result_image[result_image == np.amax(center)] = 1
                                if np.count_nonzero(result_image) < 200:
                                    cnt += 1
                                    prob_mat[patch_start_x: patch_end_x,patch_start_y: patch_end_y, l] = result_image
                            except Exception:
                                pass
                        elif clustering == 'gmm':
                            try:
                                pixels = image[image != 0]
                                vectorized = pixels.reshape((-1, 1))
                                vectorized = np.float32(vectorized)
                                gmm_model_tied = GMM(n_components=2, covariance_type='tied').fit(vectorized)
                                center_tied = gmm_model_tied.means_
                                label_tied = gmm_model_tied.predict(vectorized).reshape(-1, 1)
                                res_tied = center_tied[label_tied.flatten()]
                                result_image_tied = res_tied
                                result_image_tied[result_image_tied != np.amax(center_tied)] = 0
                                result_image_tied[result_image_tied == np.amax(center_tied)] = 1
                                b = np.zeros(img_patch.shape)
                                pos = np.where(img_patch != 0)
                                b[pos[0], pos[1]] = result_image_tied.reshape(len(img_patch[img_patch != 0]))
                                if np.count_nonzero(b) < 200:
                                    cnt += 1
                                    prob_mat[patch_start_x: patch_end_x,patch_start_y: patch_end_y, l] = b
                            except Exception:
                                pass
        # save prob_mat
        print('the number of vessel patch:', cnt)
        create_and_save_nifti(prob_mat, rough_mask_dir +
                              j.split(os.sep)[-1].split('_')[0] + '_label_rough.nii')
    print()
    print('DONE')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int,
                        help='the quadratic patch sizes.')
    parser.add_argument("--clustering", type=str,
                        help='kmeans or gmm')
    parser.add_argument("--patch_annotation_dir", type=str,
                        help='Directory for saving the images after the patch-annotation process.')
    parser.add_argument("--rough_mask_dir", type=str,
                        help='Directory for saving the images after getting rough mask.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
