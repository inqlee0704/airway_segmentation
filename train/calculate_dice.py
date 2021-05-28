import sys
sys.path.insert(0,'../../DL_code')
import os
import pandas as pd
import numpy as np
from metrics_util import Dice3d
from medpy.io import load
import time
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

start = time.time()
infer_path = r"/data4/inqlee0704/ENV18PM_ProjSubjList_IN_Inference.in"
infer_list = pd.read_csv(infer_path,sep='\t')
dices = []
for path in infer_list.ImgDir:
    pred_path = os.path.join(path,'ZUNU_unet-airtree.img.gz')
    gt_path = os.path.join(path,'ZUNU_vida-airtree.img.gz')
    print(f'Run Inference on: {path}')

    pred,_ = load(pred_path)
    gt,_ = load(gt_path)
    dice = Dice3d(pred,gt)
    dices.append(dice)
    print(dice)
print(np.mean(np.array(dices)))
end = time.time()
print('Elapsed time: ' + str(end-start))
