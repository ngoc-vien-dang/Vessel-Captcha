"""
File name: train_pnetcls.py
Author: ngocviendang
Date created: July 21, 2020

This is the script for training the pnetcls model. 
"""

import argparse
import sys
import os
import numpy as np
import nibabel as nib
import random
import pickle
from captcha.utils.pnetcls import get_pnetcls
from captcha.utils.vgg import get_vgg
from captcha.utils.unetcls import get_unetcls
from captcha.utils.resnet import get_resnet
from sklearn.preprocessing import LabelBinarizer
from captcha.utils.helper import getAllFiles

def main(args):
    patch_dir = os.path.expanduser(args.patch_dir)
    model_arch = args.model_arch
    train_metadata_filepath = args.train_metadata_filepath
    model_filepath = args.model_filepath
    unfiltered_filelist = getAllFiles(patch_dir)
    vessel_list = [item for item in unfiltered_filelist if re.search('_vessel', item)]
    empty_list = [item for item in unfiltered_filelist if re.search('_empty', item)]
    vessel_list =  sorted(vessel_list)
    empty_list = sorted(empty_list)
    train_X = np.concatenate((np.load(vessel_list[0]), np.load(empty_list[0])), axis=0)
    train_y = np.r_[1*np.ones((len(np.load(vessel_list[0])), 1)).astype('int'),
                    0*np.ones((len(np.load(empty_list[0])), 1)).astype('int')]
    for i, j in enumerate(vessel_list):
        vessel_mat = np.load(j)
        empty_mat = np.load(empty_list[i])
        if i !=0:
            train_X = np.concatenate((train_X, vessel_mat, empty_mat), axis=0)
            train_y = np.r_[train_y,
                            1*np.ones((len(vessel_mat), 1)).astype('int'),
                            0*np.ones((len(empty_mat), 1)).astype('int')]
    train_X= train_X[:, :, :, None]
    lb = LabelBinarizer()
    train_y = lb.fit_transform(train_y)
    print('Shape of train_X, train_y: ',train_X.shape, len(train_y))
    # normalize training set
    mean1 = np.mean(train_X)  # mean for data centering
    std1 = np.std(train_X)  # std for data normalization
    train_X -= mean1
    train_X /= std1
    # CREATING MODEL
    patch_size = 32
    if model_arch == 'pnetcls':
        model = get_pnetcls(patch_size)
    elif model_arch == 'vgg':
        model = get_vgg()
    elif model_arch == 'resnet':
        model = get_resnet()
    elif model_arch == 'unetcls':
        model = get_unetcls()
    elif model_arch == 'pnetcls':
        model = get_pnetcls(patch_size)
    # train model
    print('Training model...')
    model.fit(train_X, train_y, validation_split = 0.2, epochs = 10, batch_size = 64)
    # saving model
    print('Saving model to ', model_filepath)
    model.save(model_filepath)
    # saving mean and std
    print('Saving params to ', train_metadata_filepath)
    results = {'mean_train': mean1,'std_train':std1}
    with open(train_metadata_filepath, 'wb') as handle:
        pickle.dump(results, handle)
    print()
    print('DONE')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_dir", type=str,
                        help='the directory for saving extracted patches.')
    parser.add_argument("--model_arch", type=str,
                        help='vgg or resnet or pnetcls or unetcls')
    parser.add_argument("--train_metadata_filepath", type=str, 
                        help='The filename path for saving the mean and std of training set.')
    parser.add_argument("--model_filepath", type=str, help='The filename path for saving the model.')
    return parser.parse_args(argv)                                                                                 

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
