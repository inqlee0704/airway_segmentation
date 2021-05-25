# ##############################################################################
# Usage: python opening.py {subj}
# Run Time: 
# Ref: 
# ##############################################################################
# 20210521, In Kyu Lee
# Desc: Remove noises after the U-Net segmentation
# ##############################################################################
# Input: 
#  - airway mask
# Output:
#  - postprocessed airway mask
# ##############################################################################

# import libraries
import os
import sys
sys.path.append('/data1/inqlee0704/DL_code')
from model_util import get_model
# from inference import run_inference
from metrics_util import Dice3d
import time

import torch
import numpy as np
import pandas as pd
from medpy.io import load,save
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

import optuna

class CFG:
    model = 'RecursiveUNet'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/Recursive_UNet_CE_4downs_20210317/model.pth'
    infer_path = r"/data4/inqlee0704/ENV18PM_ProjSubjList_IN_Inference.in"

def run_open(subj_path,kernel=[10,10,10]):
    pred = sitk.ReadImage(subj_path)
    img = sitk.GetArrayFromImage(pred)
    img = img/255
    img = img.astype(np.uint8)
    img = sitk.GetImageFromArray(img)
    open_img = sitk.BinaryOpeningByReconstruction(img, kernel)
    open_img = sitk.GetArrayFromImage(open_img)
    return np.transpose(open_img*255,(2,1,0))

def objective(trial):
    x = trial.suggest_int('x',1,20)
    z = trial.suggest_int('z',1,20)
    kernel = [z,x,x]
    print(kernel)

    infer_list = pd.read_csv(CFG.infer_path,sep='\t')

    acc = []
    for img_dir in infer_list.ImgDir:
        print(img_dir)
        UNet_airway_path = os.path.join(img_dir,'ZUNU_unet-airtree0.img.gz')
        opened_airway = run_open(UNet_airway_path,kernel)
        VIDA_airway_path = os.path.join(img_dir,'ZUNU_vida-airtree.img.gz')
        VIDA_airway, _ = load(VIDA_airway_path)
        temp_acc = Dice3d(VIDA_airway,opened_airway)
        print(temp_acc)
        if temp_acc<0.5:
            return temp_acc
        acc.append(temp_acc)
    return np.mean(acc)



if __name__=='__main__':
    start = time.time()
    CFG = CFG()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    trial = study.best_trial

    for key, value in trial.params.items():
        print(f'{key}: {value}')
    end = time.time()
    print('Elapsed time: '+str(end-start))

    # infer_list = pd.read_csv(CFG.infer_path,sep='\t')
    # acc = []
    # kernel = [9,10,17]
    # for img_dir in infer_list.ImgDir:
    #     print(img_dir)
    #     UNet_airway_path = os.path.join(img_dir,'ZUNU_unet-airtree0.img.gz')
    #     opened_airway = run_open(UNet_airway_path,kernel)
    #     # UNet_airway, _ = load(UNet_airway_path)
    #     break
    #     VIDA_airway_path = os.path.join(img_dir,'ZUNU_vida-airtree.img.gz')
    #     VIDA_airway, hdr = load(VIDA_airway_path)
    #     temp_acc = Dice3d(VIDA_airway,opened_airway)
    #     print(temp_acc)

    #     save(opened_airway,os.path.join(img_dir,'ZUNU_unet-airtree-p_20.img.gz'),hdr=hdr)

    #     # acc.append(temp_acc)

    #     break
    # print(np.mean(acc))