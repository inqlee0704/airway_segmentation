import sys
sys.path.insert(0,'/data1/inqlee0704/DL_code')
# sys.path.insert(0,'/data1/inqlee0704/DL_code')
import torch
import pandas as pd
from inference import run_inference
from model_util import RecursiveUNet
import time
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

start = time.time()
infer_path = r"/data4/inqlee0704/ENV18PM_ProjSubjList_IN_Inference.in"
# Recursive_UNet_CE_4downs_20210310
# 20 epoch with cosine annealing warmrestart scheduler
# parameter_path = '/home/inqlee0704/airway2/RESULTS/Recursive_UNet_CE_4downs_20210310/model.pth'
# parameter_path = '/home/inqlee0704/airway2/RESULTS/Recursive_UNet_CE_4downs_20210314/model.pth'
# n_case=256
parameter_path = '/home/inqlee0704/airway2/RESULTS/Recursive_UNet_CE_4downs_20210317/model.pth'
infer_list = pd.read_csv(infer_path,sep='\t')

for path in infer_list.ImgDir:
    print(f'Run Inference on: {path}')
    run_inference(path,parameter_path)

end = time.time()
print('Elapsed time: ' + str(end-start))
