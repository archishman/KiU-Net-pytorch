# Code for KiU-Net
# Author: Jeya Maria Jose
from tqdm import tqdm
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init

from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss,classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
from functools import partial
from random import randint
import timeit

from arch.ae import kiunet,kinetwithsk,unet,autoencoder, reskiunet,densekiunet, kiunet3d

def mae(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum(abs(imageA.astype("float") - imageB.astype("float")) )
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


parser = argparse.ArgumentParser(description='KiU-Net')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
                    help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 5)

parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--load-teacher', default='default', type=str,
                    help='teacher weights')

parser.add_argument('--save', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', default='kiunet', type=str,
                    help='model name')
parser.add_argument('--direc', default='./brainus_OC_udenet', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--edgeloss', default='off', type=str)
parser.add_argument('--teacher-channel-const', default=32, type=int)
parser.add_argument('--student-channel-const', default=2, type=int)
parser.add_argument('--loss-alpha', default=0.5, type=float)

args = parser.parse_args()

aug = args.aug
direc = './RITE/results_{}_reweighted_distilled'.format(str(args.student_channel_const))
losstype= args.edgeloss
device = args.device
cuda = args.cuda
load = args.load 
load_teacher = args.load_teacher
start_epoch = args.start_epoch

def add_noise(img):
    #random noise
    noise = torch.randn(img.size()) * 0.1
    noisy_img = img + (noise.cuda() if cuda else noise)
    return noisy_img
     

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_val)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

teacher = kiunet(size = args.teacher_channel_const)


model = kiunet(size = args.student_channel_const)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()
    teacher = nn.DataParallel(teacher).cuda()
model.to(device)
teacher.to(device)


if load != 'default':
    model.load_state_dict(torch.load(load))
teacher.load_state_dict(torch.load(load_teacher))
bestdice=0
class_weights = (.0657806396484375, 0.9342193603515625)
criterion = LogNLLLoss(weight=torch.Tensor(class_weights).cuda())
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)

metric_list = MetricList({'jaccard': partial(jaccard_index),
                          'f1': partial(f1_score)})
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params), 'Loss alpha: ', args.loss_alpha)

loss_alpha = args.loss_alpha
for epoch in range(args.start_epoch,args.epochs):
    # break

    epoch_running_loss = 0
    epoch_running_teacher_loss = 0
    epoch_running_student_loss_gt = 0
    epoch_running_student_loss_teacher = 0
    for batch_idx, (X_batch, y_batch, *rest) in enumerate(tqdm(dataloader)):        
        
        ###augmentations

        X_batch = Variable(X_batch.to(device =device))
        y_batch = Variable(y_batch.to(device=device))
        
        numr = randint(0,9)
        

        if aug=='on':

            if numr == 2:

                # print(X_batch,y_batch)
                X_batch = torch.flip(X_batch,[2,3])
                y_batch = torch.flip(y_batch,[1,2])
                # print(X_batch,y_batch)
            elif numr ==3:
                X_batch = torch.flip(X_batch,[3,2])
                y_batch = torch.flip(y_batch,[2,1])
            
            elif numr==4:
                X_batch = add_noise(X_batch)
                # y_batch = add_noise(y_batch)
        
        # noisy_in = add_noise(X_batch)
        # ===================forward=====================
        output = model(X_batch)
        with torch.no_grad():
            teacher_output = teacher(X_batch)
            #print("SHAPES: ", output.shape, teacher_output.shape)
        tmp2 = teacher_output.detach().cpu().numpy()
        tmp = output.detach().cpu().numpy()
        tmp[tmp>=0.5] = 1
        tmp[tmp<0.5] = 0
        tmp2[tmp2>=0.5] = 1
        tmp2[tmp2<0.5] = 0
        tmp2 = tmp2.astype(int)
        tmp = tmp.astype(int)
        
        # print(np.unique(tmp2))
        yHaT = tmp
        yval = tmp2

        if losstype is 'on':
            edgeloss = mae(yHaT,yval)
        else:
            edgeloss = 0
        teacher_loss = criterion(teacher_output, y_batch)
        student_loss_gt = criterion(output, y_batch)
        student_loss_t =  criterion(output, teacher_output)
        loss = loss_alpha * student_loss_gt + (1 - loss_alpha) * student_loss_t
        loss =loss + edgeloss/10000
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()
        epoch_running_teacher_loss += teacher_loss.item()
        epoch_running_student_loss_gt += student_loss_gt.item()
        epoch_running_student_loss_teacher += student_loss_t.item()
        # break
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, args.epochs, epoch_running_loss/(batch_idx+1)))
    print('Breakdown - teacher_loss:{:.4f}, student_loss_gt:{:.4f}, student_loss_t:{:.4f}, '
          .format(epoch_running_teacher_loss/(batch_idx+1),
          epoch_running_student_loss_gt/(batch_idx+1),
          epoch_running_student_loss_teacher/(batch_idx+1)))
          
    # =================validation=============
    # metric_list.reset()
    tf1 = 0
    tmiou = 0
    tpa = 0
    count = 0
    if (epoch % args.save_freq) ==0:

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            # print(batch_idx)
            if isinstance(rest[0][0], str):
                        image_filename = rest[0][0]
            else:
                        image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device=device))
            y_batch = Variable(y_batch.to(device=device))
            # start = timeit.default_timer()
            y_out = model(X_batch)
            # stop = timeit.default_timer()
            # print('Time: ', stop - start) 
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
            # print(np.unique(tmp2))
            yHaT = tmp
            yval = tmp2

            epsilon = 1e-20
            
            del X_batch, y_batch,tmp,tmp2, y_out

            count = count + 1
            yHaT[yHaT==1] =255
            yval[yval==1] =255
            fulldir = direc+"/{}/".format(epoch)
            # print(fulldir+image_filename)
            if not os.path.isdir(fulldir):
                
                os.makedirs(fulldir)
            
            cv2.imwrite(fulldir+'partial_'+image_filename, tmp3[0,1,:,:])
            cv2.imwrite(fulldir+image_filename, yHaT[0,1,:,:])

            # cv2.imwrite(fulldir+'/gt_{}.png'.format(count), yval[0,:,:])
        fulldir = direc+"/{}/".format(epoch)
        torch.save(model.state_dict(), fulldir+args.model+".pth")
        torch.save(model.state_dict(), direc+"model.pth")
            
        if bestdice<tf1:
            bestdice = tf1 
            print("bestdice = {}".format(bestdice/count))  
            print(epoch) 

