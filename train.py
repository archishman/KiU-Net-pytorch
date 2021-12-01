# Code for KiU-Net
# Author: Jeya Maria Jose
import argparse
import sys
import os
import timeit
from functools import partial
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data as data
import torchvision
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from arch.ae import kiunet
from metrics import LogNLLLoss, classwise_f1, f1_score, jaccard_index
from utils import (Image2D, ImageToImage2D, JointTransform2D, Logger,
                   MetricList, chk_mkdir)

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)

def mae(imageA, imageB):
    err = np.sum(abs(imageA.astype("float") - imageB.astype("float")) )
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err
def get_parser():
    parser = argparse.ArgumentParser(description='KiU-Net')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                        help='Number of training epochs')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='M',
                        help='Manual epoch number (useful on restarts)')
    parser.add_argument('-w', '--weights', default=None, type=str,
                        help='Path to pretrained weights')
    parser.add_argument('--size', default='default', type=int,
                        help='Number of starting channels in KiU-Net')
    parser.add_argument('--learning-rate', default=1e-4, type=float,
                        help='Learning Rate')
    parser.add_argument('--train_dataset', required=True, type=str)
    parser.add_argument('--val_dataset', type=str)
    parser.add_argument('--save_freq', type=int,default = 5)
    parser.add_argument('--direc', default= None, type=str, help='directory to save')
    parser.add_argument('--crop', type=int, default=None)
    return parser

def clean_args(args):
    args.direc = args.direc or './RITE/results_{}_reweighted'.format(str(args.size))
    return args

def get_data_loaders(train_dataset, val_dataset):
    tf_train = JointTransform2D(crop=None, p_flip=0, p_random_affine=0, color_jitter_params=None, long_mask=True)
    tf_val = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(train_dataset, tf_train)
    val_dataset = ImageToImage2D(val_dataset, tf_val)
    predict_dataset = Image2D(args.val_dataset)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valloader = DataLoader(val_dataset, 1, shuffle=True)
    return trainloader, valloader

def train(model, epoch_range, trainloader, valloader, criterion, optimizer, metric_list, ):
    bestdice=0
    def train_epoch(epoch):
        epoch_running_loss = 0
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(tqdm(trainloader)):        
            ###augmentations
            X_batch = Variable(X_batch.cuda())
            y_batch = Variable(y_batch.cuda())
            
            # ===================forward=====================
            output = model(X_batch)
            
            tmp2 = y_batch.detach().cpu().numpy()
            tmp = output.detach().cpu().numpy()
            tmp[tmp>=0.5] = 1
            tmp[tmp<0.5] = 0
            tmp2[tmp2>0] = 1
            tmp2[tmp2<=0] = 0
            tmp2 = tmp2.astype(int)
            tmp = tmp.astype(int)
            yHaT = tmp
            yval = tmp2

            edgeloss = mae(yHaT,yval)

            loss = criterion(output, y_batch) #+ mae(yHaT,yval)/10000
            
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_running_loss += loss.item()
        return epoch_running_loss/(batch_idx+1)
    def validate():
        tf1 = 0
        tmiou = 0
        tpa = 0
        count = 0
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            if isinstance(rest[0][0], str):
                        image_filename = rest[0][0]
            else:
                        image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.cuda())
            y_batch = Variable(y_batch.cuda())
            y_out = model(X_batch)
            tmp2 = y_batch.detach().cpu().numpy()
            tmp = y_out.detach().cpu().numpy()
            tmp3 = np.copy(tmp)
            tmp3[tmp3<0] = 0
            tmp3[tmp3>1] = 1
            tmp3 = tmp3 * 255 
            tmp[tmp>=0.5] = 1
            tmp[tmp<0.5] = 0
            tmp2[tmp2>0] = 1
            tmp2[tmp2<=0] = 0
            tmp2 = tmp2.astype(int)
            tmp = tmp.astype(int)
            tmp3 = tmp3.astype(int) 
            yHaT = tmp
            yval = tmp2

            epsilon = 1e-20
            
            del X_batch, y_batch,tmp,tmp2, y_out

            count = count + 1
            yHaT[yHaT==1] =255
            yval[yval==1] =255
            fulldir = args.direc+"/{}/".format(epoch)
            if not os.path.isdir(fulldir):
                
                os.makedirs(fulldir)
            
            cv2.imwrite(fulldir+'partial_'+image_filename, tmp3[0,1,:,:])
            cv2.imwrite(fulldir+image_filename, yHaT[0,1,:,:])

            fulldir = args.direc+"/{}/".format(epoch)
            torch.save(model.state_dict(), fulldir+"kiunet.pth")
            torch.save(model.state_dict(), args.direc+"model.pth")
        return tf1

        
    for epoch in epoch_range:
        epoch_loss = train_epoch(epoch)
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch, max(epoch_range), epoch_loss))
        
        # =================validation=============
        # metric_list.reset()
        
        if (epoch % args.save_freq) ==0:
            
            tf1 = validate()    
            if bestdice<tf1:
                bestdice = tf1 
                print("bestdice = {}".format(bestdice/count))  
                print(epoch) 


if __name__ == "__main__":
    args = clean_args(get_parser().parse_args())
    model = kiunet(size = args.size)
    # model.apply(weight_init)
    model = nn.DataParallel(model).cuda()
    if args.weights:
        model.load_state_dict(torch.load(args.weights))
    trainloader, valloader = get_data_loaders(args.train_dataset, args.val_dataset)
    class_weights = (.0657806396484375, 0.9342193603515625)
    criterion = LogNLLLoss(weight=torch.Tensor(class_weights).cuda())
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                                weight_decay=1e-5)

    metric_list = MetricList({'jaccard': partial(jaccard_index),
                            'f1': partial(f1_score)})
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    train(model, range(args.start_epoch, args.epochs), trainloader, valloader, criterion, optimizer, metric_list)
    