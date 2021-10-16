import torch
from torch import nn
import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from medpy.io import load,save

# from UNet import RecursiveUNet
from custom_UNet import UNet
import nibabel as nib
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


start = time.time()
infer_path = r"/home/inqlee0704/src/TE16/ProjSubjList.in"
# infer_path = r"/data4/inqlee0704/ENV18PM_ProjSubjList_IN_Inference.in"
# parameter_path = '/home/inqlee0704/airway2/RESULTS/Recursive_UNet_CE_4downs_20210317/model.pth'
# th=0.4: 0.8896, relu
# parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/UNet_baseline_relu_20210814/airway_UNet.pth'
# th=0.4, 0.9192
# parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/UNet_BCE_dice_ncase0_20210819/airway_UNet.pth'
# th
parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/MultiC_UNet_20210921/airway_UNet_multiC.pth'
infer_list = pd.read_csv(infer_path,sep='\t')
result_path = '/data4/inqlee0704/TE/airway_mask/multiC_UNet_th1e-5'
if not os.path.exists(result_path):
    os.mkdir(result_path)
# Model
# model = RecursiveUNet(num_classes=1, activation=nn.ReLU(inplace=True))
# model = RecursiveUNet(num_classes=1, activation=nn.LeakyReLU(inplace=True))
model = UNet(in_channel=3)
model.load_state_dict(torch.load(parameter_path))
DEVICE = 'cuda'
model.to(DEVICE)
model.eval()
th = 1e-5

# ths = np.arange(0.3,0.6,0.1)
# for th in ths:
    # Each subject
dice = []
pbar = tqdm(infer_list.ImgDir,total=len(infer_list))
for subj_path in pbar:
    # print(subj_path)
    # volume, hdr = load(os.path.join(subj_path,'zunu_vida-ct.img'))
    volume, hdr = load(subj_path)
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
    hdr = nib.Nifti1Header()
    pair_img = nib.Nifti1Pair(slices,np.eye(4),hdr)
    nib.save(pair_img,result_path+'/'+str(subj_path[-9:-7])+'.img.gz')


end = time.time()
print('Elapsed time: ' + str(end-start))
