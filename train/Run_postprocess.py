import os
import sys
sys.path.insert(0,'/data1/inqlee0704/DL_code')
import pandas as pd
from medpy.io import load,save
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import gzip
import shutil
import time
import numpy as np

start = time.time()


infer_path = r"/data4/inqlee0704/ENV18PM_ProjSubjList_IN_Inference.in"
infer_list = pd.read_csv(infer_path,sep='\t')

c = 3
for path in infer_list.ImgDir:
    print(path)
    pred,hdr = load(os.path.join(path,'ZUNU_unet-airtree.img.gz'))
    # Make it binary
    pred = pred/255
    pred = pred.astype(np.uint8)
    (x,y,z) = hdr.get_voxel_spacing()
    kernel = [round(c/x),round(c/y),round(c/z)]
    img = sitk.GetImageFromArray(pred)
    open_img = sitk.BinaryOpeningByReconstruction(img, kernel)
    final_img = sitk.GetArrayFromImage(open_img)*255
    # Rename ZUNU_unet-airtree -> ZUNU_unet_airtree0
    print('Rename ZUNU_unet-airtree.* -> ZNUN_unet-airtree0.*')
    os.rename(os.path.join(path,'ZUNU_unet-airtree.img.gz'),
                os.path.join(path,'ZUNU_unet-airtree0.img.gz'))
    os.rename(os.path.join(path,'ZUNU_unet-airtree.hdr'),
                os.path.join(path,'ZUNU_unet-airtree0.hdr'))
    
    # Save post-processed airway mask
    save(final_img,os.path.join(path,'ZUNU_unet-airtree.img.gz'),hdr=hdr)
    with gzip.open(os.path.join(path,'ZUNU_unet-airtree.hdr.gz'),'rb') as f_in:
        with open(os.path.join(path,'ZUNU_unet-airtree.hdr'),'wb') as f_out:
            shutil.copyfileobj(f_in,f_out)
    os.remove(os.path.join(path,'ZUNU_unet-airtree.hdr.gz'))
    
end = time.time()
print(f'Elapsed time: {end-start}')
