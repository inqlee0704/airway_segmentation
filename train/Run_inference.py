import torch
from torch import nn
import os

import numpy as np
import pandas as pd
from medpy.io import load,save

from UNet import RecursiveUNet
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
parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/UNet_baseline_relu_20210814/airway_UNet.pth'
infer_list = pd.read_csv(infer_path,sep='\t')

# Model
model = RecursiveUNet(num_classes=1, activation=nn.ReLU(inplace=True))
model.load_state_dict(torch.load(parameter_path))
DEVICE = 'cuda'
model.to(DEVICE)
model.eval()
# th = 0.5

ths = np.arange(0.2,0.8,0.1)

for th in ths:
    # Each subject
    dice = []
    for subj_path in infer_list.ImgDir:
        print(subj_path)
        volume, hdr = load(os.path.join(subj_path,'zunu_vida-ct.img'))
        mask, _ = load(os.path.join(subj_path,'ZUNU_vida-airtree.img.gz'))
        volume = (volume-np.min(volume)) / (np.max(volume)-np.min(volume))

        # Each slice
        slices = np.zeros(volume.shape)
        for z in range(volume.shape[2]):
            s = volume[:,:,z]
            s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
            pred = model(s.to(DEVICE,dtype=torch.float))
            pred = torch.sigmoid(pred)
            pred = np.squeeze(pred.cpu().detach())
            slices[:,:,z] = (pred>th).float()
            # break
        dice_ = Dice3d(slices,mask)
        print(dice_)
        dice.append(dice_)
    print('--------------------')
    print(th)
    print(f'Mean: {np.mean(dice)}')
    print('--------------------')



end = time.time()
print('Elapsed time: ' + str(end-start))
