# ##############################################################################
# Usage: python inference.py {CaseID} post
# python inference.py 62
# Run Time: 
# Ref: 
# ##############################################################################
# 20210521, In Kyu Lee
# Desc: 2D U-Net airway segmentation
# combined VIDA segmenation + UNet segmentation
# ##############################################################################
# Input: 
#  - CT analyze file
#  - post
# Output:
#  - airway mask analyze file
# ##############################################################################
import sys
sys.path.insert(0,'E:\\common\InKyu\\DL_code')
from inference import volume_inference
from model_util import get_model
sys.path.insert(1,'../util')
from DCM2IMG import DCMtoVidaCT
import torch
import numpy as np
import os
import time
from medpy.io import save, load
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import shutil # copy file
import gzip # unzip .hdr.gz -> .hdr

#---------------------------------------------------------------------------------------
CaseID = str(sys.argv[1])
# CaseID = '66'
class CFG:
    model = 'RecursiveUNet'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    parameter_path = 'E:\\common\\InKyu\\airway_segmentation\\train\\RESULTS\\model.pth'
    root_path = os.path.join('E:\\VIDA\\VIDAvision2.2',CaseID)
    save_path = os.path.join('E:\\VIDA_more\\UNet',CaseID)
    NoEdit_path = os.path.join('E:\\VIDA_more\\NoEdit',CaseID)
#---------------------------------------------------------------------------------------
CFG = CFG()
#---------------------------------------------------------------------------------------
# 0. Copy initial VIDA to NoEdit
if not os.path.exists(CFG.NoEdit_path):
    print(f'{CaseID} is not copied to NoEdit')
    print(f'Copying {CaseID} Folder to NoEdit/{CaseID}. . .')
    shutil.copytree(CFG.root_path, CFG.NoEdit_path) 

# 1. Create save folder
if not os.path.exists(CFG.save_path):
    print(f'Creating {CFG.save_path}')
    os.mkdir(CFG.save_path)

# 2. Create ANALYZE file from DICOM
if not os.path.exists(os.path.join(CFG.root_path,'zunu_vida-ct.img')):
    print(f'No zunu_vida-ct.img file found')
    print(f'Creating ANALYZE file from DICOM. . .')
    DCMtoVidaCT(CFG.root_path,CFG.save_path)
else:
    print('zunu_vida-ct.img file found!')
    print(f'Move zunu_vida-ct: {CFG.root_path} --> {CFG.save_path}')
    os.rename(os.path.join(CFG.root_path,'ZUNU_vida-airtree.img.gz'),os.path.join(CFG.save_path,'ZUNU_vida-airtree.img.gz'))
    os.rename(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr'),os.path.join(CFG.save_path,'ZUNU_vida-airtree.hdr'))

# 3. Rename & Move 
# VIDA -> VIDA_more\UNet
# - ZUNU_vida-airtree.img.gz -> ZUNU_vida-airtree_0.img.gz
# - ZUNU_vida-airtree.hdr -> ZUNU_vida-airtree_0.hdr
print('Rename ZUNU_vida-airtree.* -> ZUNU_vida-airtree_0.*')
os.rename(os.path.join(CFG.root_path,'ZUNU_vida-airtree.img.gz'),os.path.join(CFG.save_path,'ZUNU_vida-airtree_0.img.gz'))
os.rename(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr'),os.path.join(CFG.save_path,'ZUNU_vida-airtree_0.hdr'))

# 4. U-Net: airway segmentation
start = time.time()

## load VIDA processed mask
print('Data Loading . . .')
mask_path = os.path.join(CFG.save_path,'ZUNU_vida-airtree_0.img.gz')
vida_img, label = load(mask_path)

## load CT image
img_path = os.path.join(CFG.save_path,'zunu_vida-ct.img')
image,_ = load(img_path)
image = (image-(np.min(image)))/((np.max(image)-(np.min(image))))
out = []
out.append({'image':image})
test_data = np.array(out)

## load model
model = get_model(CFG)
model.load_state_dict(torch.load(CFG.parameter_path))
model.to(CFG.DEVICE)
model.eval()
print('Start Inference . . .')
kernel = [5,5,5]
for x in test_data:
    pred = volume_inference(model,x['image'])
    pred = pred*255
    pred = pred.astype(np.ubyte)
    save(pred,os.path.join(CFG.save_path,'ZUNU_unet-airtree.img.gz'),hdr=label)
    
    # combine vida process & Unet
    combined = (vida_img==255) | (pred==255)
    combined = combined.astype(np.ubyte)

    # Postprocess
    combined = sitk.GetImageFromArray(combined)
    open_img = sitk.BinaryOpeningByReconstruction(combined, kernel)
    open_img = sitk.GetArrayFromImage(open_img)
    open_img = open_img*255
    open_img = open_img.astype(np.ubyte)

    save(open_img,os.path.join(CFG.root_path,'ZUNU_vida-airtree.img.gz'),hdr=label)
end = time.time()

print('Elapsed time: ' + str(end-start))

# 5. Unzip .hdr.gz -> .hdr
with gzip.open(os.path.join(CFG.save_path,'ZUNU_unet-airtree.hdr.gz'),'rb') as f_in:
    with open(os.path.join(CFG.save_path,'ZUNU_unet-airtree.hdr'),'wb') as f_out:
        shutil.copyfileobj(f_in,f_out)
os.remove(os.path.join(CFG.save_path,'ZUNU_unet-airtree.hdr.gz'))

with gzip.open(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr.gz'),'rb') as f_in:
    with open(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr'),'wb') as f_out:
        shutil.copyfileobj(f_in,f_out)
os.remove(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr.gz'))

# 6. Copy 
# VIDA -> VIDA_more\UNet
# - ZUNU_vida-airtree.img.gz -> ZUNU_unet-airtree.img.gz
# - ZUNU_vida-airtree.hdr -> ZUNU_unet-airtree.hdr
print('Copy ZUNU_vida-airtree.* -> ZUNU_unet-airtree.*')
shutil.copyfile(os.path.join(CFG.root_path,'ZUNU_vida-airtree.img.gz'),os.path.join(CFG.save_path,'ZUNU_comb-airtree.img.gz'))
shutil.copyfile(os.path.join(CFG.root_path,'ZUNU_vida-airtree.hdr'),os.path.join(CFG.save_path,'ZUNU_comb-airtree.hdr'))
