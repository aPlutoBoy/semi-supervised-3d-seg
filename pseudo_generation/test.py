import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk

import random
import argparse
import time
import json
import numpy as np
import torch

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm
import neurite as ne

# dice loss computation
def Dice_loss(pred, true):
    top = np.sum(2 * pred * true)
    bottom = np.sum(pred) + np.sum(true)
    dice = top / bottom
    return dice

# part of the mri_3d 
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
    atlas[patient]['head'] = head
    atlas[patient]['re_atlas'] = abs_atlas - head
    return mri_3d_part

# data loading and preprocessing
def dataloader():
    print("now start loading data and just waiting...")
    dir_path = 'MICCAI_BraTS2020_TrainingData'
    files = os.listdir(dir_path)
    test_data = dict()
    test_seg = []
    test_flair = []
    test_t1 = []
    test_t2 = []
    test_t1ce = []
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
                test_seg.append(mri_3d_pad != 0)            
            elif not os.path.isdir(gz_path) and sequence == 'flair':
                test_flair.append(mri_3d_pad)
            elif not os.path.isdir(gz_path) and sequence == 't1':
                test_t1.append(mri_3d_pad)
            elif not os.path.isdir(gz_path) and sequence == 't2':
                test_t2.append(mri_3d_pad)
            elif not os.path.isdir(gz_path) and sequence == 't1ce':
                test_t1ce.append(mri_3d_pad)
        i = i + 1
        if np.mod(i, 100) == 0:
            print("finish loading %d patients already!" % i)
    test_data['seg'] = test_seg
    test_data['flair'] = test_flair
    test_data['t1'] = test_t1
    test_data['t2'] = test_t2
    test_data['t1ce'] = test_t1ce
    
    return test_data, atlas

# voxelmorph data generator
def vxm_val_generator(test_data, atlas):
    
    patients = len(test_data['seg'])
    depth = 80

    sequence = {0:'flair',
                1:'t1', 
                2:'t1ce', 
                3:'t2'}

    for patient in range(patients):
        if np.mod(patient, 100) == 0 and patient != 0:
            print("already generate pseudo mask for %d patients" % patient)
        for d in range(depth):
            src = []
            trg = []
            for i in range(4):
                data = test_data[sequence[i]][patient][atlas[patient]['re_atlas']][np.newaxis, ..., np.newaxis]
                src.append(data)
                data = test_data[sequence[i]][patient][d][np.newaxis, ..., np.newaxis]
                trg.append(data)
            mask = test_data['seg'][patient][atlas[patient]['re_atlas']][np.newaxis, ..., np.newaxis]
            yield (src, trg, mask, patient, d)
    
    return False

# using trained model generate pseudo mask
def pseudo_generation(model_path):
    base_path = 'MICCAI_Pseudo_Mask_zcq'
    os.makedirs(base_path, exist_ok=True)
    
    test_data, atlas = dataloader()
 
    generator = vxm_val_generator(test_data=test_data, atlas=atlas)
    print("finish generating val generator ")

    sequence = {0:'flair',
                1:'t1', 
                2:'t1ce', 
                3:'t2'}

    device = 'cuda'
    model = vxm.networks.VxmDense.load(model_path, device)
    model.to(device)
    model.eval()
    
    while True:
        src, trg, mask, patient, depth = next(generator)
        gt_mask = test_data['seg'][patient][depth]
        src = [torch.from_numpy(c).to(device).float().permute(0, 3, 1, 2) for c in src]
        trg = [torch.from_numpy(c).to(device).float().permute(0, 3, 1, 2) for c in trg]
        mask = torch.from_numpy(mask).to(device).float().permute(0, 3, 1, 2)

        for i in range(len(src)):

            # predict
            inputs = [src[i], trg[i]]
            moved, warp = model(*inputs, registration=True)

            # using flow map to generate pseudo mask
            transformer = vxm.layers.SpatialTransformer((256, 256))
            transformer.to(device)
            pseudo_mask = transformer(mask, warp)
            pseudo_mask = pseudo_mask.detach().cpu().numpy().squeeze()

            # pseudo mask saving
            img_base_path = os.path.join(base_path, '%03d' % (patient+1), sequence[i])
            os.makedirs(img_base_path, exist_ok=True)
            cv2.imwrite(img_base_path+'/%03d.jpg' % (depth+atlas[patient]['head']), pseudo_mask*255)
            #dice = Dice_loss(pseudo_mask, gt_mask)


if __name__ == "__main__":
    
    model_path = '1500.pt'

    pseudo_generation(model_path)