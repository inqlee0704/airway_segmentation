# ##############################################################################
# Usage: python inference.py {CaseID} post
# python inference.py 62
# Run Time:
# Ref:
# ##############################################################################
# 20211210, In Kyu Lee
# Desc: Ensemble 2D U-Net airway segmentation
# model1 + model2
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


# ---------------------------------------------------------------------------------------
CaseID = str(sys.argv[1])
# CaseID = '66'
class CFG:
    # model = 'RecursiveUNet'
    model = "MultiC_ZUNet"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    parameter_path = "E:\\common\\InKyu\\airway_segmentation\\train\\RESULTS\\model.pth"
    root_path = os.path.join("E:\\VIDA\\VIDAvision2.2", CaseID)
    save_path = os.path.join("E:\\VIDA_more\\DL", CaseID)
    NoEdit_path = os.path.join("E:\\VIDA_more\\NoEdit", CaseID)


# ---------------------------------------------------------------------------------------
CFG = CFG()
# ---------------------------------------------------------------------------------------
# 0. Copy initial VIDA to NoEdit
if not os.path.exists(CFG.NoEdit_path):
    print(f"{CaseID} is not copied to NoEdit")
    print(f"Copying {CaseID} Folder to NoEdit/{CaseID}. . .")
    shutil.copytree(CFG.root_path, CFG.NoEdit_path)

# 1. Create save folder
if not os.path.exists(CFG.save_path):
    print(f"Creating {CFG.save_path}")
    os.mkdir(CFG.save_path)

# 2. Create ANALYZE file from DICOM
if not os.path.exists(os.path.join(CFG.root_path, "zunu_vida-ct.img")):
    print(f"No zunu_vida-ct.img file found")
    print(f"Creating ANALYZE file from DICOM. . .")
    DCMtoVidaCT(CFG.root_path, CFG.save_path)
else:
    print("zunu_vida-ct.img file found!")
    print(f"Move zunu_vida-ct: {CFG.root_path} --> {CFG.save_path}")
    os.rename(
        os.path.join(CFG.root_path, "ZUNU_vida-airtree.img.gz"),
        os.path.join(CFG.save_path, "ZUNU_vida-airtree.img.gz"),
    )
    os.rename(
        os.path.join(CFG.root_path, "ZUNU_vida-airtree.hdr"),
        os.path.join(CFG.save_path, "ZUNU_vida-airtree.hdr"),
    )

# 3. Rename & Move
# VIDA -> VIDA_more\UNet
# - ZUNU_vida-airtree.img.gz -> ZUNU_vida-airtree_0.img.gz
# - ZUNU_vida-airtree.hdr -> ZUNU_vida-airtree_0.hdr
print("Rename ZUNU_vida-airtree.* -> ZUNU_vida-airtree_0.*")
os.rename(
    os.path.join(CFG.root_path, "ZUNU_vida-airtree.img.gz"),
    os.path.join(CFG.save_path, "ZUNU_vida-airtree_0.img.gz"),
)
os.rename(
    os.path.join(CFG.root_path, "ZUNU_vida-airtree.hdr"),
    os.path.join(CFG.save_path, "ZUNU_vida-airtree_0.hdr"),
)

# 4. U-Net: airway segmentation
start = time.time()

## load VIDA processed mask
print("Data Loading . . .")
mask_path = os.path.join(CFG.save_path, "ZUNU_vida-airtree_0.img.gz")
vida_img, label = load(mask_path)

## load CT image
img_path = os.path.join(CFG.save_path, "zunu_vida-ct.img")
image, _ = load(img_path)
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
    pred = volume_inference(model, x["image"])
    pred = pred * 255
    pred = pred.astype(np.ubyte)
    save(pred, os.path.join(CFG.save_path, "ZUNU_zunet-airtree.img.gz"), hdr=label)

    # combine vida process & Unet
    combined = (vida_img == 255) | (pred == 255)
    combined = combined.astype(np.ubyte)

    # Postprocess
    combined = sitk.GetImageFromArray(combined)
    open_img = sitk.BinaryOpeningByReconstruction(combined, kernel)
    open_img = sitk.GetArrayFromImage(open_img)
    open_img = open_img * 255
    open_img = open_img.astype(np.ubyte)

    save(open_img, os.path.join(CFG.root_path, "ZUNU_vida-airtree.img.gz"), hdr=label)
end = time.time()

print("Elapsed time: " + str(end - start))

# 5. Unzip .hdr.gz -> .hdr
with gzip.open(os.path.join(CFG.save_path, "ZUNU_zunet-airtree.hdr.gz"), "rb") as f_in:
    with open(os.path.join(CFG.save_path, "ZUNU_zunet-airtree.hdr"), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove(os.path.join(CFG.save_path, "ZUNU_zunet-airtree.hdr.gz"))

with gzip.open(os.path.join(CFG.root_path, "ZUNU_vida-airtree.hdr.gz"), "rb") as f_in:
    with open(os.path.join(CFG.root_path, "ZUNU_vida-airtree.hdr"), "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
os.remove(os.path.join(CFG.root_path, "ZUNU_vida-airtree.hdr.gz"))

# 6. Copy
# VIDA -> VIDA_more\UNet
# - ZUNU_vida-airtree.img.gz -> ZUNU_zunet-airtree.img.gz
# - ZUNU_vida-airtree.hdr -> ZUNU_zunet-airtree.hdr
print("Copy ZUNU_vida-airtree.* -> ZUNU_zunet-airtree.*")
shutil.copyfile(
    os.path.join(CFG.root_path, "ZUNU_vida-airtree.img.gz"),
    os.path.join(CFG.save_path, "ZUNU_comb-airtree.img.gz"),
)
shutil.copyfile(
    os.path.join(CFG.root_path, "ZUNU_vida-airtree.hdr"),
    os.path.join(CFG.save_path, "ZUNU_comb-airtree.hdr"),
)
