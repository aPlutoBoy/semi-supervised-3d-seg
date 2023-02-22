import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'

import matplotlib.pyplot as plt
import SimpleITK as sitk

import random
import json
import argparse
import shutil
import time
import numpy as np
import torch

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

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
    atlas[patient]['re_atlas'] = abs_atlas - head
    return mri_3d_part

# data loading and preprocessing
def dataloader():
    print("now start loading data and just waiting...")
    dir_path = 'MICCAI_BraTS2020_TrainingData'
    files = os.listdir(dir_path)
    train_data = dict()
    train_flair = []
    train_t1 = []
    train_t2 = []
    train_t1ce = []
    pad_amount = ((0,0), (8,8), (8,8))
    atlas = json.load(open('atlas.json', 'r'))
    i = 0
    for file in files:
        # if i > 10:
        #     break
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
            if not os.path.isdir(gz_path) and sequence == 'flair':
                train_flair.append(mri_3d_pad)
            elif not os.path.isdir(gz_path) and sequence == 't1':
                train_t1.append(mri_3d_pad)
            elif not os.path.isdir(gz_path) and sequence == 't2':
                train_t2.append(mri_3d_pad)
            elif not os.path.isdir(gz_path) and sequence == 't1ce':
                train_t1ce.append(mri_3d_pad)
        i = i + 1
        if np.mod(i, 100) == 0:
            print("finish loading %d patients already!" % i)
    train_data['flair'] = train_flair
    train_data['t1'] = train_t1
    train_data['t2'] = train_t2
    train_data['t1ce'] = train_t1ce
    
    return train_data, atlas

# voxelmorph data generator
def vxm_data_generator(train_data, atlas, batch_size=8):
    
    vol_shape = train_data['t1'][0].shape[1:]
    vol_depth = train_data['t1'][0].shape[0]
    dims = len(vol_shape)
    num = len(train_data['t1'])

    zeros = np.zeros([batch_size, *vol_shape, dims])
    sequence = dict()
    sequence[0] = 'flair'
    sequence[1] = 't1'
    sequence[2] = 't1ce'
    sequence[3] = 't2'

    while True:
        src_vol = []
        trg_vol = []
        s = np.random.randint(0, 4)
        idx1 = random.sample(range(0, num), batch_size)
        idx2 = np.random.randint(0, vol_depth, size=batch_size)
        data = train_data[sequence[s]]
        for i in range(batch_size):
            src_vol.append(data[idx1[i]][atlas[idx1[i]]['re_atlas']][np.newaxis, ..., np.newaxis])
            trg_vol.append(data[idx1[i]][idx2[i]][np.newaxis, ..., np.newaxis])
        
        src_vol =np.concatenate(src_vol, axis=0)
        trg_vol = np.concatenate(trg_vol, axis=0)

        in_vols = [src_vol, trg_vol]
        out_vols = [trg_vol, zeros]

        yield (in_vols, out_vols)


def adjust_learning_rate(optimizer, epoch, epochs):
    lr = np.logspace(-4, -5, num=epochs)[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, model_dir):
    filename = os.path.join(model_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth'))

def train(generator, model, losses, optimizer, epoch, device, steps_per_epoch=100, loss_file=None):

    model.train()

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]
            
        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)

        # calculate total loss
        loss = 0
        loss_list = []

        curr_loss = losses[0](y_true[0], y_pred[0]) 
        loss_list.append(curr_loss.item())
        loss += curr_loss
        curr_loss = losses[1](y_true[0], y_pred[0]) * 0.01
        loss_list.append(curr_loss.item())
        loss += curr_loss  

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    with open(loss_file, 'a+') as f:
        f.write(' - '.join((epoch_info, time_info, loss_info)))
        f.write('\n')

    return np.mean(epoch_total_loss)

def main(initial_epoch, epochs, loss, batch_size, load_model=None):
    
    best_loss = 1e6

    train_data, atlas = dataloader()
    print("data loading complete!")

    generator = vxm_data_generator(train_data=train_data, atlas=atlas, batch_size=batch_size)
    print('data generator done!')

    # model defination
    # extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1]

    # prepare model folder
    model_dir = 'noatt_best_model_' + str(epochs) + '_' + loss
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    # gpus = args.gpu.split(',')
    device = 'cuda'
    nb_gpus = torch.cuda.device_count()
    print('GPU NUM: {:2d}'.format(nb_gpus))
    assert np.mod(batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (batch_size, nb_gpus)


    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = True

    # unet architecture
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]

    # load pre-trained model
    if load_model != None:
        model = vxm.networks.VxmDense.load(load_model, device)
    else:
        model = vxm.networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=None,
            int_steps=7,
            int_downsize=2,
            src_feats=1,
            trg_feats=1,
            attention=False
            )

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save
    
    model.to(device)

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        image_loss_func = vxm.losses.NCC().loss

    # need two image loss functions if bidirectional
    losses = [image_loss_func]
    weights = [1]

    # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=2).loss]
    weights += [0.01]

    loss_file = model_dir + '/' + str(epochs) + '_' + loss + '.txt'

    for epoch in range(initial_epoch, epochs):
        adjust_learning_rate(optimizer, epoch, epochs)

        # train for one epoch
        loss = train(generator=generator, model=model, losses=losses, optimizer=optimizer, device=device, epoch=epoch, loss_file=loss_file)

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        if is_best:
            model.save(os.path.join(model_dir, '%04d.pt' % epochs))
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'best_loss': best_loss,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, model_dir)

if __name__ == "__main__":
    
    # training settings
    initial_epoch = 0
    epochs = 1500
    loss = 'ncc'
    load_model = None
    batch_size = 40

    main(initial_epoch=initial_epoch, epochs=epochs, loss=loss, batch_size=batch_size, load_model=load_model)