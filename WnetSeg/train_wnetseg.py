"""
File name: train_wnetseg.py
Author: ngocviendang
Date created: July 13, 2020

This is the script for training the wnetseg model. 
"""
import argparse
import sys
import pickle
import time
import numpy as np
import re
import os 
from keras.optimizers import Adam
from WnetSeg.utils.metrics import dice_coef_loss, dice_coef
from WnetSeg.utils.helper import gen_filename_pairs_2
from WnetSeg.utils.wnetseg import get_wnetseg
from WnetSeg.utils.unet import get_unet
from WnetSeg.utils.pnet import get_pnet
from keras.preprocessing.image import ImageDataGenerator

def main(args):
    skull_stripping_dir = os.path.expanduser(args.skull_stripping_dir)
    patch_size = args.patch_size
    model_arch = args.model_arch
    train_metadata_filepath = args.train_metadata_filepath
    model_filepath = args.model_filepath
    input_list, label_list = gen_filename_pairs_2(skull_stripping_dir, 'img', 'label')
    input_list = sorted(input_list)
    label_list = sorted(label_list)
    train_X = np.load(input_list[0])
    train_y = np.load(label_list[0])
    for i, j in enumerate(input_list):
        img_mat = np.load(j)
        label_mat = np.load(label_list[i])
        if i !=0:
            train_X = np.concatenate((train_X, img_mat), axis=0)
            train_y = np.concatenate((train_y, label_mat), axis=0)
    train_X, train_y = train_X[:, :, :, None], train_y[:, :, :, None]
    # normalize data: mean and std calculated on the train set and applied to the train and val set
    mean1 = np.mean(train_X)  # mean for data centering
    std1 = np.std(train_X)  # std for data normalization
    train_X -= mean1
    train_X /= std1
    print(train_X.shape)
    print(train_y.shape)
    # set hyperparameters
    batch_size = 64
    num_channels = 1
    activation = 'relu'
    final_activation = 'sigmoid'
    optimizer = Adam
    lr = 1e-4
    dropout = 0.1
    num_epochs = 10
    loss = dice_coef_loss
    metrics = [dice_coef, 'accuracy']
    # CREATING MODEL
    if model_arch == 'wnetseg':
        model = get_wnetseg(patch_size, num_channels, activation,
                            final_activation, optimizer, lr, dropout, loss, metrics)
    elif model_arch == 'unet':
        model = get_unet(patch_size, num_channels, activation,
                         final_activation, optimizer, lr, dropout, loss, metrics)
    elif model_arch == 'pnet':
        model = get_pnet(patch_size, optimizer, lr,
                         loss, metrics, num_channels)
    # CREATING DATAGENTORATOR
    # how many times to augment training samples with the ImageDataGenerator per one epoch
    factor_train_samples = 2
    rotation_range = 30  # for ImageDataGenerator
    horizontal_flip = False  # for ImageDataGenerator
    vertical_flip = True  # for ImageDataGenerator
    shear_range = 20  # for ImageDataGenerator
    width_shift_range = 0  # for ImageDataGenerator
    height_shift_range = 0  # for ImageDataGenerator
    data_gen_args = dict(rotation_range=rotation_range,
                         horizontal_flip=horizontal_flip,
                         vertical_flip=vertical_flip,
                         shear_range=shear_range,
                         width_shift_range=width_shift_range,
                         height_shift_range=height_shift_range,
                         fill_mode='constant')
    X_datagen = ImageDataGenerator(**data_gen_args)
    y_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    X_datagen.fit(train_X, augment=True, seed=seed)
    y_datagen.fit(train_y, augment=True, seed=seed)

    X_generator = X_datagen.flow(
        train_X, batch_size=batch_size, seed=seed, shuffle=True)
    y_generator = y_datagen.flow(
        train_y, batch_size=batch_size, seed=seed, shuffle=True)

    # combine generators into one which yields image and label
    train_generator = zip(X_generator, y_generator)
    # TRAINING MODEL
    start_train = time.time()
    model.fit_generator(train_generator,
                                  steps_per_epoch=factor_train_samples *
                                  len(train_X) // batch_size,
                                  epochs=num_epochs,
                                  verbose=2, shuffle=True)
    duration_train = int(time.time() - start_train)
    print('training took:', (duration_train // 3600) % 60, 'hours', (duration_train // 60) % 60,
          'minutes', duration_train % 60, 'seconds')
    # saving model
    print('Saving model to ', model_filepath)
    model.save(model_filepath)
    # saving mean and std
    print('Saving params to ', train_metadata_filepath)
    results = {'mean_train': mean1, 'std_train': std1}
    with open(train_metadata_filepath, 'wb') as handle:
        pickle.dump(results, handle)
    print()
    print('DONE')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--skull_stripping_dir", type=str,
                        help='Directory for saving the images after the skull stripping process.')
    parser.add_argument("--model_arch", type=str,
                       help='pnet or unet or wnetseg.')
    parser.add_argument("--train_metadata_filepath", type=str,
                        help='The filename path for saving the mean and std of training set.')
    parser.add_argument("--model_filepath", type=str,
                        help='The filename path for saving the model.')
    parser.add_argument("--patch_size", type=int,
                        help='the quadratic patch sizes.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))






