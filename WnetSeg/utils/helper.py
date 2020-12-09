"""
File name: helper.py
Author: ngocviendang
Date created: July 13, 2020

This file contains helper functions for other scripts.
"""
import os
import re
import nibabel as nib
import numpy as np
import random

def listfiles(folder):
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            yield os.path.join(root, filename)

def gen_filename_pairs_2(data_dir, v_re, l_re):
    unfiltered_filelist = list(listfiles(data_dir))
    input_list = [item for item in unfiltered_filelist if re.search(v_re, item)]
    label_list = [item for item in unfiltered_filelist if re.search(l_re, item)]
    print(input_list)
    print(label_list)
    print("input_list size:    ", len(input_list))
    print("label_list size:    ", len(label_list))
    if len(input_list) != len(label_list):
        print("input_list size and mask_list and label_list size don't match")
        raise Exception
    return input_list, label_list


def gen_filename_pairs_3(data_dir, v_re, m_re, l_re):
    unfiltered_filelist = list(listfiles(data_dir))
    input_list = [item for item in unfiltered_filelist if re.search(v_re, item)]
    mask_list = [item for item in unfiltered_filelist if re.search(m_re, item)]
    label_list = [item for item in unfiltered_filelist if re.search(l_re, item)]
    print(input_list)
    print(mask_list)
    print(label_list)
    print("input_list size:    ", len(input_list))
    print("mask_list size:    ", len(mask_list))
    print("label_list size:    ", len(label_list))
    if len(input_list) != len(label_list) or len(input_list) != len(mask_list):
        print("input_list size and mask_list and label_list size don't match")
        raise Exception
    return input_list, mask_list, label_list

def load_nifti_mat_from_file(path_orig):
    """
    # Reference https://github.com/prediction2020/unet-vessel-segmentation.git
    Loads a nifti file and returns the data from the nifti file as numpy array.
    :param path_orig: String, path from where to load the nifti.
    :return: Nifti data as numpy array.
    """
    nifti_orig = nib.load(path_orig)
    print(' - nifti loaded from:', path_orig)
    print(' - dimensions of the loaded nifti: ', nifti_orig.shape)
    print(' - nifti data type:', nifti_orig.get_data_dtype())
    return nifti_orig.get_data()  # transform the images into np.ndarrays


def create_and_save_nifti(mat, path_target):
    """
    # Reference https://github.com/prediction2020/unet-vessel-segmentation.git
    Creates a nifti image from numpy array and saves it to given path.
    :param mat: Numpy array.
    :param path_target: String, path where to store the created nifti.
    """
    new_nifti = nib.Nifti1Image(mat, np.eye(4))  # create new nifti from matrix
    nib.save(new_nifti, path_target)  # save nifti to target dir
    print('New nifti saved to:', path_target)

def Rand(start, end, num):
    res = []
    for j in range(num):
        res.append(random.randint(start, end))
    return res

def aplly_mask(mat, mask_mat):
    """
    # Reference https://github.com/prediction2020/unet-vessel-segmentation.git
    Masks the image with the given mask.
    :param mat: Numpy array, image to be masked.
    :param mask_mat: Numpy array, mask.
    :return: Numpy array, masked image.
    """
    masked = mat
    masked[np.where(mask_mat == 0)] = 0
    return masked
