"""
File name: evaluation_segmentation.py
Author: ngocviendang
Date created: July 21, 2020

This file compute f1-score 
"""
import pickle
import numpy as np
import os
import argparse
import sys
from captcha.utils.helper import getAllFiles
from captcha.utils.helper import load_nifti_mat_from_file
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from captcha.utils.metrics import avg_class_acc


def main(args):
    result_dir = args.result_dir  
    # List filenames of data after the skull stripping process
    unfiltered_filelist = getAllFiles(result_dir)
    label_list = [item for item in unfiltered_filelist if re.search('_label', item)]
    prediction_list = [item for item in unfiltered_filelist if re.search('_prediction', item)]
    label_list = sorted(label_list)
    prediction_list = sorted(prediction_list)
    print(label_list)
    print(prediction_list)
    acc_list = []
    f1_score_list = []
    # load image, mask and label stacks as matrices
    for i, j in enumerate(label_list):
        print('> Loading label...')
        label_mat = load_nifti_mat_from_file(j)
        print('> Loading prediction...')
        prediction_mat = load_nifti_mat_from_file(prediction_list[i])
        pred_class = prediction_mat
        converted_pred_class = pred_class.copy().astype(int)
        converted_pred_class[converted_pred_class == 1] = -1
        converted_pred_class[converted_pred_class == 0] = 1
        print()
        #print("Performance of patient: ", j.split(os.sep)[-1].split('_')[0])
        label_mat_f = label_mat.flatten()
        #prob_mat_f = prob_mat.flatten()
        pred_class_f = pred_class.flatten().astype(np.uint8)

        #pat_auc = roc_auc_score(label_mat_f, prob_mat_f)
        pat_acc = accuracy_score(label_mat_f, pred_class_f)
        pat_avg_acc, tn, fp, fn, tp = avg_class_acc(label_mat_f, pred_class_f)
        pat_dice = f1_score(label_mat_f, pred_class_f)
        acc_list.append(pat_acc*100)
        f1_score_list.append(pat_dice*100)
        print('acc:', round(pat_acc,4))
        print('avg acc:', pat_avg_acc)
        print('dice:', round(pat_dice,4))
        print()
    print('mean of acc: ', round(np.mean(acc_list),2))
    print('std of acc: ', round(np.std(acc_list),2))
    print('mean of f1 score: ', round(np.mean(f1_score_list),2))
    print('std of f1 score: ', round(np.std(f1_score_list),2))
    print('DONE')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str,
                        help='the filename path of predictions.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


