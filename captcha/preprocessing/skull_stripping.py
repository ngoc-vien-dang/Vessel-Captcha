"""
File name: skull_stripping.py
Author: ngocviendang
Date created: July 13, 2020

This file removes the skull from the MRI images.
"""
import argparse
import sys
import os
import numpy as np
import nibabel as nib
from captcha.utils import helper

def main(args):
    original_data_dir = os.path.expanduser(args.original_data_dir)
    target_dir = os.path.expanduser(args.target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # List filenames of data
    #input_list, mask_list, label_list = helper.gen_filename_pairs_3(original_data_dir, 'ToF', 'mask', 'vessel')
    input_list, mask_list, label_list = helper.gen_filename_pairs_3(original_data_dir, 'img', 'mask', 'grid')
    input_list = sorted(input_list)
    mask_list = sorted(mask_list)
    label_list = sorted(label_list)
    print(input_list)
    print(mask_list)
    print(label_list)
    # load image, mask and label stacks as matrices	
    for i,j in enumerate(input_list):
        print('Loading image...')
        img_mat = helper.load_nifti_mat_from_file(j)
        print('Loading mask...')
        mask_mat = helper.load_nifti_mat_from_file(mask_list[i])
        print('Loading label...')
        label_mat = helper.load_nifti_mat_from_file(label_list[i])      
        # check the dimensions
        assert img_mat.shape == mask_mat.shape == label_mat.shape, "The DIMENSIONS of image, mask and label are NOT " \
                                                                   "SAME."

        # mask images and labels (skull stripping)
        img_mat = helper.aplly_mask(img_mat, mask_mat)
        label_mat = helper.aplly_mask(label_mat, mask_mat)
        print(j.split(os.sep)[-1].split('_')[0])
        # save to new file as masked version of original data 
        helper.create_and_save_nifti(img_mat, target_dir + j.split(os.sep)[-1].split('_')[0] + '_img.nii')
        helper.create_and_save_nifti(mask_mat, target_dir + j.split(os.sep)[-1].split('_')[0] + '_mask.nii')
        helper.create_and_save_nifti(label_mat, target_dir + j.split(os.sep)[-1].split('_')[0] + '_label.nii')
        
        print()
    print('DONE')
   
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data_dir", type=str,
                        help='data dictionary.')
    parser.add_argument("--target_dir", type=str, help='Directory for saving the images after the skull stripping process.')     
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

