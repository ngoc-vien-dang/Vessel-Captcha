"""
File name: predict_full_testset.py
Author: ngocviendang
Date created: July 13, 2020

This file for predicting segmentations of the whole test set.
"""
import pickle
import time
import numpy as np
import os
import argparse
import sys
import nibabel as nib
from keras.models import load_model
from captcha.utils.metrics import dice_coef_loss, dice_coef
from captcha.utils.helper import create_and_save_nifti
from captcha.utils.helper import load_nifti_mat_from_file
from captcha.utils.helper import getAllFiles

def main(args):
    test_set_dir = args.test_set_dir
    patch_size = args.patch_size
    model_arch = args.model_arch
    train_metadata_filepath = args.train_metadata_filepath
    model_filepath = args.model_filepath
    prediction_filepath = args.prediction_filepath
    # LOADING MODEL, RESULTS AND WHOLE BRAIN MATRICES
    print(model_filepath)
    model = load_model(model_filepath, custom_objects={
                       'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    with open(train_metadata_filepath, 'rb') as handle:
        train_metadata = pickle.load(handle)
    print(train_metadata)
    # List filenames of data after the skull stripping process
    unfiltered_filelist = getAllFiles(test_set_dir)
    input_list = [item for item in unfiltered_filelist if re.search('_img', item)]
    mask_list = [item for item in unfiltered_filelist if re.search('_mask', item)]
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
        # prediction
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
        starttime_total = time.time()
        # proceed slice by slice
        for l in z_slices:
            print('Slice:', l)
            starttime_slice = time.time()
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
            num_of_x_patches = np.int(
                np.ceil((slice_x_max - slice_x_min) / patch_size))
            num_of_y_patches = np.int(
                np.ceil((slice_y_max - slice_y_min) / patch_size))
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
                    img_patch -= train_metadata['mean_train']
                    img_patch /= train_metadata['std_train']

                    # predict the patch with the model and save to probability matrix
                    prob_mat[patch_start_x: patch_end_x, patch_start_y: patch_end_y, l] = np.reshape(
                        model.predict(np.reshape(
                        img_patch, (1, patch_size, patch_size, 1)), batch_size=1, verbose=0),
                        (patch_size, patch_size))

            # how long does the prediction take for one slice
            duration_slice = time.time() - starttime_slice
            print('prediction in slice took:', (duration_slice // 3600) % 60, 'hours',
              (duration_slice // 60) % 60, 'minutes',
              duration_slice % 60, 'seconds')
        # how long does the prediction take for a patient
        duration_total = time.time() - starttime_total
        print('prediction in total took:', (duration_total // 3600) % 60, 'hours',
          (duration_total // 60) % 60, 'minutes',
          duration_total % 60, 'seconds')
        # save file
        print(j.split(os.sep)[-1].split('_')[0])
        if model_arch == 'wnetseg':
            create_and_save_nifti(prob_mat, prediction_filepath +'prediction' +'_'+
                                    j.split(os.sep)[-1].split('_')[0] + '_wnetseg.nii.gz')
        elif model_arch == 'pnet':
            create_and_save_nifti(prob_mat, prediction_filepath +'prediction' +'_'+
                                    j.split(os.sep)[-1].split('_')[0] + '_pnet.nii.gz')
        elif model_arch == 'unet':
            create_and_save_nifti(prob_mat, prediction_filepath +'prediction' +'_'+
                                    j.split(os.sep)[-1].split('_')[0] + '_unet.nii.gz')
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set_dir", type=str,
                        help='the filename path of the test set.')
    parser.add_argument("--model_arch", type=str,
                        help='pnet or unet or wnetseg.')
    parser.add_argument("--patch_size", type=int,
                        help='patch_size.')
    parser.add_argument("--train_metadata_filepath", type=str,
                        help='The filename path for loading the mean and std of training set.')
    parser.add_argument("--model_filepath", type=str,
                        help='The filename path for loading the model.')
    parser.add_argument("--prediction_filepath", type=str,
                        help='The filename path for saving the prediction.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
