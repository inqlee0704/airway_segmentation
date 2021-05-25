# ##############################################################################
# Usage: python inference.py {CaseID} post
# python inference.py 62
# Run Time: 
# Ref: 
# ##############################################################################
# 20210521, In Kyu Lee
# Desc: 2D U-Net airway segmentation
# ##############################################################################
# Input: 
#  - CT analyze file
#  - post
# Output:
#  - airway mask analyze file
# ##############################################################################
import sys
# sys.path.append('E:\\common\\InKyu\\src-master\\DL')
sys.path.insert(0,'/data1/inqlee0704/DL_code')
from engine import volume_inference
from model_util import get_model
sys.path.insert(1,'../util')
from DCM2IMG import DCMtoVidaCT
import torch
import numpy as np
import os
import time
from mepy.io import save, load
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import shutil # copy file
import gzip # unzip .hdr.gz -> .hdr


#---------------------------------------------------
class CFG:
    model = 'RecursiveUNet'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/Recursive_UNet_CE_4downs_20210317/model.pth'
    # parameter_path = 'E:\\common\\InKyu\\src\\model.pth'
    root_path = os.path.join('E:\\VIDA\\VIDAvision2.2',CaseID)
    save_path = os.path.join('E:\\VIDA_more\\UNet',CaseID)
#---------------------------------------------------
CaseID = str(sys.argv[1])
# CaseID = '66'
CFG = CFG()
#----------------------------------------
# 0. Create save folder
if not os.path.exists(CFG.save_path):
    print(f'Creating {CFG.save_path}')
    os.mkdir(CFG.save_path)

# 1. Create ANALYZE file from DICOM
if not os.path.exists(os.path.join(CFG.root_path,'zunu_vida-ct.img')):
    print(f'No zunu_vida-ct.img file found')
    print(f'Create ANALYZE file from DICOM. . .')
    DCMtoVidaCT(CFG.root_path,CFG.save_path)
else:
    print('zunu_vida-ct.img file found!')
    print(f'Move zunu_vida-ct: {CFG.root_path} --> {CFG.save_path}')
    os.rename(os.path.join(CFG.root_path,'ZUNU_vida-airtree.img.gz'),os.path.join(CFG.save_path,'ZUNU_vida-airtree.img.gz'))
    os.rename(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr'),os.path.join(CFG.save_path,'ZUNU_vida-airtree.hdr'))

# 2. Rename & Move 
# VIDA -> VIDA_more\UNet
# - ZUNU_vida-airtree.img.gz -> ZUNU_vida-airtree_0.img.gz
# - ZUNU_vida-airtree.hdr -> ZUNU_vida-airtree_0.hdr
print('Rename ZUNU_vida-airtree.* -> ZUNU_vida-airtree_0.*')
os.rename(os.path.join(CFG.root_path,'ZUNU_vida-airtree.img.gz'),os.path.join(CFG.save_path,'ZUNU_vida-airtree_0.img.gz'))
os.rename(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr'),os.path.join(CFG.save_path,'ZUNU_vida-airtree_0.hdr'))

# 3. U-Net: airway segmentation
start = time.time()

# load header file from the VIDA processed mask
print('Data Loading . . .')
mask_path = os.path.join(CFG.save_path,'ZUNU_vida-airtree_0.img.gz')
_, label = load(mask_path)

# load CT image
img_path = os.path.join(CFG.save_path,'zunu_vida-ct.img')
image,_ = load(img_path)
image = (image-(np.min(image)))/((np.max(image)-(np.min(image))))
out = []
out.append({'image':image})
test_data = np.array(out)

# load model
model = get_model(CFG)
model.load_state_dict(torch.load(CFG.parameter_path))
model.to(CFG.DEVICE)
model.eval()
print('Start Inference . . .')
kernel = [1,9,9]
for x in test_data:
    pred_label = volume_inference(model,x['image'])
    pred_label = pred_label.astype(np.ubyte)
    # Postprocess
    img = sitk.GetImageFromArray(pred_label)
    open_img = sitk.BinaryOpeningByReconstruction(img, kernel)
    open_img = sitk.GetArrayFromImage(open_img)
    open_img = np.transpose(open_img*255,(2,1,0))
    save(open_img,os.path.join(CFG.root_path,'ZUNU_vida-airtree.img.gz'),hdr=label)
end = time.time()

print('Elapsed time: ' + str(end-start))

# 4. Unzip .hdr.gz -> .hdr
with gzip.open(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr.gz'),'rb') as f_in:
    with open(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr'),'wb') as f_out:
        shutil.copyfileobj(f_in,f_out)
os.remove(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr.gz'))

# 6. Copy 
# VIDA -> VIDA_more\UNet
# - ZUNU_vida-airtree.img.gz -> ZUNU_unet-airtree.img.gz
# - ZUNU_vida-airtree.hdr -> ZUNU_unet-airtree.hdr
print('Copy ZUNU_vida-airtree.* -> ZUNU_unet-airtree.*')
shutil.copyfile(os.path.join(CFG.root_path,'ZUNU_vida-airtree.img.gz'),os.path.join(CFG.save_path,'ZUNU_unet-airtree.img.gz'))
shutil.copyfile(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr'),os.path.join(CFG.save_path,'ZUNU_unet-airtree.hdr'))