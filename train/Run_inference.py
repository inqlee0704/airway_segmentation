import torch
from torch import nn
import os

import numpy as np
import pandas as pd
from medpy.io import load,save

# from UNet import RecursiveUNet
from custom_UNet import UNet
from ZUNet_v1 import ZUNet_v1
import segmentation_models_pytorch as smp

import time
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

def Dice3d(a,b):
    # print(f'pred shape: {a.shape}')
    # print(f'target shape: {b.shape}')
    intersection =  np.sum((a!=0)*(b!=0))
    volume = np.sum(a!=0) + np.sum(b!=0)
    if volume == 0:
        return -1
    return 2.*float(intersection)/float(volume)

def get_3Channel(img):
    img[img<-1024] = -1024
    airway_c = np.copy(img)
    tissue_c = np.copy(img)
    airway_c[airway_c>=-800] = -800
    tissue_c[tissue_c<=-200] = -200
    tissue_c[tissue_c>=200] = 200
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    tissue_c = (tissue_c-np.min(tissue_c))/(np.max(tissue_c)-np.min(tissue_c))
    airway_c = (airway_c-np.min(airway_c))/(np.max(airway_c)-np.min(airway_c))
    airway_c = airway_c[None,:]
    tissue_c = tissue_c[None,:]
    img = img[None,:]
    img_3 = np.concatenate([img,airway_c,tissue_c],axis=0)
    return img_3



# Recursive_UNet_CE_4downs_20210310
# 20 epoch with cosine annealing warmrestart scheduler
# parameter_path = '/home/inqlee0704/airway2/RESULTS/Recursive_UNet_CE_4downs_20210310/model.pth'
# parameter_path = '/home/inqlee0704/airway2/RESULTS/Recursive_UNet_CE_4downs_20210314/model.pth'
# n_case=256
# parameter_path = '/home/inqlee0704/airway2/RESULTS/Recursive_UNet_CE_4downs_20210317/model.pth'
# parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/Recursive_UNet_CE_4downs_20210526/airway_UNet.pth'



start = time.time()
infer_path = r"/data4/inqlee0704/ENV18PM_ProjSubjList_IN_Inference.in"
# parameter_path = '/home/inqlee0704/airway2/RESULTS/Recursive_UNet_CE_4downs_20210317/model.pth'
# th=0.4: 0.8896, relu
# parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/UNet_baseline_relu_20210814/airway_UNet.pth'
# th=0.4, 0.9192
# parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/UNet_BCE_dice_ncase0_20210819/airway_UNet.pth'
# th=0.3, 0.9166
# parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/MultiC_UNet_20210921/airway_UNet_multiC.pth'
parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/ZUNet_v1_multiC_20210927/airway_UNet.pth'

infer_list = pd.read_csv(infer_path,sep='\t')

# Model
# model = RecursiveUNet(num_classes=1, activation=nn.ReLU(inplace=True))
# model = RecursiveUNet(num_classes=1, activation=nn.LeakyReLU(inplace=True))
# model = UNet(in_channel=1)
# model = UNet(in_channel=3)
model = ZUNet_v1(in_channels=3)
model.load_state_dict(torch.load(parameter_path))
DEVICE = 'cuda'
model.to(DEVICE)
model.eval()
# th = 0.5

ths = np.arange(0.1,0.2,0.1)

for th in ths:
    # Each subject
    dice = []
    for subj_path in infer_list.ImgDir:
        print(subj_path)
        volume, hdr = load(os.path.join(subj_path,'zunu_vida-ct.img'))
        mask, _ = load(os.path.join(subj_path,'ZUNU_vida-airtree.img.gz'))
        # volume = (volume-np.min(volume)) / (np.max(volume)-np.min(volume))
        volume = get_3Channel(volume)

        # Each slice
        # slices = np.zeros(volume.shape)
        slices = np.zeros((512,512,volume.shape[3]))
        for z in range(volume.shape[3]):
            # s = volume[:,:,z]
            # s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
            s = volume[:,:,:,z]
            s = torch.from_numpy(s).unsqueeze(0)
            pred = model(s.to(DEVICE,dtype=torch.float))
            pred = torch.sigmoid(pred)
            pred = np.squeeze(pred.cpu().detach())
            slices[:,:,z] = (pred>th).float()
            # break
        dice_ = Dice3d(slices,mask)
        print(dice_)
        dice.append(dice_)
    print('--------------------')
    print(f'Threshold: {th}')
    print(f'Mean: {np.mean(dice)}')
    print('--------------------')

end = time.time()
print('Elapsed time: ' + str(end-start))
