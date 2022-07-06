from turtle import forward
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys
import random
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader
import math
import cv2 as cv

from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d

import torch.optim as optim

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class DoubleConv(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(DoubleConv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size = 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size = 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.model(x)

class Unet(nn.Module):
    def __init__(self,output_features):
        super(Unet, self).__init__()
        
        # Down
        self.down = nn.MaxPool2d(2)

        self.d_conv1 = DoubleConv(3,64)
        self.d_conv2 = DoubleConv(64,128)
        self.d_conv3 = DoubleConv(128,256)
        self.d_conv4 = DoubleConv(256,512)
        # # Up
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   
        self.u_conv1 = DoubleConv(256+512,256)
        self.u_conv2 = DoubleConv(256+128,128)
        self.u_conv3 = DoubleConv(64+128,64)


        self.last = nn.Conv2d(64,1,kernel_size=1)

    def forward(self,x):
        # Down
        # print("Input",x.shape)
        conv_1 = self.d_conv1(x)
        x = self.down(conv_1)
        # print("Down1",x.shape)
        conv_2 = self.d_conv2(x)
        x = self.down(conv_2)
        # print("Down2",x.shape)


        conv_3 = self.d_conv3(x)
        x = self.down(conv_3)
        conv_4 = self.d_conv4(x)
       

       
        x = self.up(conv_4)    
        x = torch.cat([x, conv_3],axis=1)
        x = self.u_conv1(x)
        x = self.up(x)

        # print(x.shape,conv_2.shape)
        x = torch.cat([x,conv_2],axis=1)
        x = self.u_conv2(x)
        x = self.up(x)
        x = torch.cat([x,conv_1],axis=1)
        x = self.u_conv3(x)

        x = self.last(x)
        return x



def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))



class UNet2(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
                
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.conv_up3 = double_conv(256 + 512, 256)
        self.conv_up2 = double_conv(128 + 256, 128)
        self.conv_up1 = double_conv(128 + 64, 64)
        
        self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
        
    def forward(self, x):
        # Batch - 1d tensor.  N_channels - 1d tensor, IMG_SIZE - 2d tensor.
        # Example: x.shape >>> (10, 3, 256, 256).
        conv1 = self.conv_down1(x)  # <- BATCH, 3, IMG_SIZE  -> BATCH, 64, IMG_SIZE..
        x = self.maxpool(conv1)     # <- BATCH, 64, IMG_SIZE -> BATCH, 64, IMG_SIZE 2x down.
        conv2 = self.conv_down2(x)  # <- BATCH, 64, IMG_SIZE -> BATCH,128, IMG_SIZE.
        x = self.maxpool(conv2)     # <- BATCH, 128, IMG_SIZE -> BATCH, 128, IMG_SIZE 2x down.
        
        
        conv3 = self.conv_down3(x)  # <- BATCH, 128, IMG_SIZE -> BATCH, 256, IMG_SIZE.
        x = self.maxpool(conv3)     # <- BATCH, 256, IMG_SIZE -> BATCH, 256, IMG_SIZE 2x down.
        x = self.conv_down4(x)      # <- BATCH, 256, IMG_SIZE -> BATCH, 512, IMG_SIZE.
        x = self.upsample(x)        # <- BATCH, 512, IMG_SIZE -> BATCH, 512, IMG_SIZE 2x up.
        #(Below the same)                                 N this       ==        N this.  Because the first N is upsampled.
        x = torch.cat([x, conv3], dim=1) # <- BATCH, 512, IMG_SIZE & BATCH, 256, IMG_SIZE--> BATCH, 768, IMG_SIZE.
        
        x = self.conv_up3(x) #  <- BATCH, 768, IMG_SIZE --> BATCH, 256, IMG_SIZE. 
        x = self.upsample(x)  #  <- BATCH, 256, IMG_SIZE -> BATCH,  256, IMG_SIZE 2x up.   
        x = torch.cat([x, conv2], dim=1) # <- BATCH, 256,IMG_SIZE & BATCH, 128, IMG_SIZE --> BATCH, 384, IMG_SIZE.  

        x = self.conv_up2(x) # <- BATCH, 384, IMG_SIZE --> BATCH, 128 IMG_SIZE. 
        x = self.upsample(x)   # <- BATCH, 128, IMG_SIZE --> BATCH, 128, IMG_SIZE 2x up.     
        x = torch.cat([x, conv1], dim=1) # <- BATCH, 128, IMG_SIZE & BATCH, 64, IMG_SIZE --> BATCH, 192, IMG_SIZE.  
        
        x = self.conv_up1(x) # <- BATCH, 128, IMG_SIZE --> BATCH, 64, IMG_SIZE.
        
        out = self.last_conv(x) # <- BATCH, 64, IMG_SIZE --> BATCH, n_classes, IMG_SIZE.
        out = torch.sigmoid(out)
        
        return out

if __name__ == "__main__":
    # image_random_raw = cv.imread('/Users/christianpederjacobsen/Dropbox/Mac/Desktop/leg/brain_cancer_seg/data/archive/kaggle_3m/TCGA_FG_5962_20000626/TCGA_FG_5962_20000626_2_mask.tif')
    # image_random = np.moveaxis(image_random_raw, -1, 0)
    image_random = np.random.normal(0,1, (20, 3, 256, 256)).astype('float32')
    image_random = Variable(torch.from_numpy(image_random))
    # conv = DoubleConv(3,64)
    # output = conv(image_random)
    # print(output.shape)
    # print(image_random.shape)
    # plt.imshow(image_random_raw)
    
    # test = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    # print(test(image_random).shape)
    net = Unet(1)
    print(net.parameters())
    print("Number of parameters in model:", get_n_params(net))
    print(net(image_random))

