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

# sys.path.insert(0, "E:\\common\InKyu\\DL_code")
# from inference import volume_inference
# from model_util import get_model

sys.path.insert(1, "../util")
from DCM2IMG import DCMtoVidaCT
import torch
import numpy as np
import os
import time
from medpy.io import save, load
import SimpleITK as sitk
from ZUNet_v1 import ZUNet_v1

# from preactivation_UNet import UNet
from engine import Segmentor

sitk.ProcessObject_SetGlobalWarningDisplay(False)
import shutil  # copy file
import gzip  # unzip .hdr.gz -> .hdr


def get_3Channel(img):
    img[img < -1024] = -1024
    airway_c = np.copy(img)
    tissue_c = np.copy(img)
    airway_c[airway_c >= -800] = -800
    tissue_c[tissue_c <= -200] = -200
    tissue_c[tissue_c >= 200] = 200
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    tissue_c = (tissue_c - np.min(tissue_c)) / (np.max(tissue_c) - np.min(tissue_c))
    airway_c = (airway_c - np.min(airway_c)) / (np.max(airway_c) - np.min(airway_c))
    airway_c = airway_c[None, :]
    tissue_c = tissue_c[None, :]
    img = img[None, :]
    img_3 = np.concatenate([img, airway_c, tissue_c], axis=0)
    return img_3


def volume_inference(model, volume, th=0.5):
    DEVICE = "cuda"
    slices = np.zeros((512, 512, volume.shape[3]))
    for z in range(volume.shape[3]):
        # s = volume[:,:,z]
        # s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        s = volume[:, :, :, z]
        s = torch.from_numpy(s).unsqueeze(0)
        pred = model(s.to(DEVICE, dtype=torch.float))
        pred = torch.sigmoid(pred)
        pred = np.squeeze(pred.cpu().detach())
        slices[:, :, z] = (pred > th).float()
    return slices

def volume_inference_z(model, volume, th=0.5):
    DEVICE = "cuda"
    volume = get_3Channel(volume)
    # slices = np.zeros(volume.shape)
    slices = np.zeros((512,512,volume.shape[3]))
    for z in range(volume.shape[3]):
        # s = volume[:,:,z]
        # s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        s = volume[:,:,:,z]
        s = torch.from_numpy(s).unsqueeze(0)
        z_frac = z/(volume.shape[3]+1)
        # z ranges from 0 to 9
        z_frac = np.floor(z_frac*10)
        z_frac = torch.tensor(z_frac, dtype=torch.int64)
        pred = model(s.to(DEVICE,dtype=torch.float), z_frac.to(DEVICE))
        pred = torch.sigmoid(pred)
        pred = np.squeeze(pred.cpu().detach())
        slices[:,:,z] = (pred>th).float()*255
    return slices
# ---------------------------------------------------------------------------------------
CaseID = str(sys.argv[1])
# CaseID = '66'
class CFG:
    # model = 'RecursiveUNet'
    model = "MultiC_ZUNet"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # parameter_path = "E:\\common\\InKyu\\airway_segmentation\\train\\RESULTS\\model.pth"
    parameter_path = '/data1/inqlee0704/airway_segmentation/train/RESULTS/ZUNet_v1_multiC_20210927/airway_UNet.pth'
    root_path = os.path.join('/data4/common/ImageData/DCM_C19_MULTI_EXPIRATION',CaseID)
    
    # save_path = os.path.join("E:\\VIDA_more\\DL", CaseID)
    # NoEdit_path = os.path.join("E:\\VIDA_more\\NoEdit", CaseID)


# ---------------------------------------------------------------------------------------
CFG = CFG()
# ---------------------------------------------------------------------------------------

# 2. Create ANALYZE file from DICOM
if not os.path.exists(os.path.join(CFG.root_path, "zunu_vida-ct.img")):
    print(f"No zunu_vida-ct.img file found")
    print(f"Creating ANALYZE file from DICOM. . .")
    DCMtoVidaCT(CFG.root_path, CFG.root_path)
else:
    print("zunu_vida-ct.img file found!")

# 4. U-Net: airway segmentation
start = time.time()

## load CT image
img_path = os.path.join(CFG.root_path, "zunu_vida-ct.img")
image, label = load(img_path)
multic_img = get_3Channel(image)
# image = (image - (np.min(image))) / ((np.max(image) - (np.min(image))))
out = []
# out.append({"image": image})
out.append({"image": multic_img})
test_data = np.array(out)

## load model
model = ZUNet_v1(in_channels=3)
model.load_state_dict(torch.load(CFG.parameter_path))
model.to(CFG.DEVICE)
model.eval()
print("Start Inference . . .")
kernel = [5, 5, 5]
for x in test_data:
    pred = volume_inference_z(model, x["image"])
    pred = pred * 255
    pred = pred.astype(np.ubyte)
    save(pred, os.path.join(CFG.root_path, "ZUNU_zunet-airtree.img.gz"), hdr=label)

end = time.time()

print("Elapsed time: " + str(end - start))

