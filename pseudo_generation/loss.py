import os, sys

import matplotlib.pyplot as plt
import SimpleITK as sitk

import cv2
import random
import json
import argparse
import shutil
import time
import numpy as np

# dice loss defination
def Dice_loss(pred, true):

    top = np.sum(2 * pred * true)
    bottom = np.sum(pred) + np.sum(true)
    if bottom == 0:
        dice = 1
    else:
        dice = top / bottom

    return dice

# cover compute
def overlap(pred, true):

    top = np.sum(pred * true)
    bottom = np.sum(true)
    if bottom == 0:
        cover = None
    else:
        cover = top / bottom
    return cover

def cut(mri_3d, atlas, patient):
    abs_atlas = atlas[patient]['atlas']
    head = abs_atlas - 40
    tail = abs_atlas + 40
    if head < 0:
        head = 0
        tail = 80
    if tail >= 155:
        tail = 154
        head = 74
    mri_3d_part = mri_3d[head : tail, ...]
    atlas[patient]['re_atlas'] = abs_atlas - head
    return mri_3d_part

def dataloader(dir_path='MICCAI_BraTS2020_TrainingData'):
    print("now start loading data and just waiting...")
    files = os.listdir(dir_path)
    gt_mask = []
    pad_amount = ((0,0), (8,8), (8,8))
    atlas = json.load(open('atlas.json', 'r'))
    i = 0
    for file in files:
        if i > 31:
            break
        file_path = os.path.join(dir_path, file)
        gzs = os.listdir(file_path)
        for gz in gzs:
            gz_path = os.path.join(file_path, gz)
            sequence = gz_path.split('.')[0].split('_')[-1]
            mri_3d = sitk.ReadImage(gz_path)
            mri_3d = sitk.Cast(sitk.RescaleIntensity(mri_3d), sitk.sitkUInt8)
            mri_3d = sitk.GetArrayFromImage(mri_3d)
            mri_3d_part = cut(mri_3d=mri_3d, atlas=atlas, patient=i)
            mri_3d_pad = np.pad(mri_3d_part, pad_amount, 'constant')
            mri_3d_pad = mri_3d_pad.astype('float') / 255
            if not os.path.isdir(gz_path) and sequence == 'seg':
                gt_mask.append(mri_3d_pad != 0)
        i = i + 1
        if np.mod(i, 100) == 0:
            print("finish loading %d patients already!" % i)

    return gt_mask

def pseudoloader(dir_path='MICCAI_Pseudo_Mask_noatt'):
    print("now start loading pseudo mask and just waiting...")
    files = os.listdir(dir_path)
    pseudo_mask = []

    i = 0
    for file in files:
        file_path = os.path.join(dir_path, file)
        sequences = os.listdir(file_path)
        s = dict()
        for sequence in sequences:
            sequence_path = os.path.join(file_path, sequence)
            imgs = os.listdir(sequence_path)
            mask = [cv2.imread(os.path.join(sequence_path, img), 0).astype('float') / 255 for img in imgs]
            mask = [a[np.newaxis, ...] for a in mask]
            mask = np.concatenate(mask, axis=0)
            s[sequence] = mask
        pseudo_mask.append(s)
        i = i + 1
    
    return pseudo_mask


def compute(pred, true):
    print("now start computing and just waiting...")
    num = len(true)
    s = {0: 'flair', 1: 't1', 2: 't1ce', 3: 't2'}
    result = []
    for i in range(num):
        gt_mask = true[i]
        pseudo_mask = pred[i]
        depth = true[i].shape[0]
        m = dict()
        for j in range(4):
            m[s[j]] = dict()
            dice = [Dice_loss(pseudo_mask[s[j]][d], gt_mask[d]) for d in range(depth)]
            while 0 in dice:
                dice.remove(0)
            avg_dice = np.mean(dice)
            var_dice = np.var(dice)
            cover = [overlap(pseudo_mask[s[j]][d],gt_mask[d]) for d in range(depth)]
            while None in cover:
                cover.remove(None)
            avg_cover = np.mean(cover)
            var_cover = np.var(cover)
            m[s[j]]['avg_dice'] = avg_dice
            m[s[j]]['var_dice'] = var_dice
            m[s[j]]['avg_cover'] = avg_cover
            m[s[j]]['var_cover'] = var_cover
        result.append(m)
    json.dump(result, open('result.json', 'w'), indent=4)

def main():
    gt_mask = dataloader()
    pseudo_mask = pseudoloader()
    compute(pseudo_mask, gt_mask)

if __name__ == "__main__":
    main()