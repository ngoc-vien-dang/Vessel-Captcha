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

def getAllFiles(dir, result = None):
    if result is None:
        result = []
    for entry in os.listdir(dir):
        entrypath = os.path.join(dir, entry)
        if os.path.isdir(entrypath):
            getAllFiles(entrypath ,result)
        else:
            result.append(entrypath)
    result = sorted(result)
    return result

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
