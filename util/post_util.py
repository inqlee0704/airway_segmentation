# ##############################################################################
# Usage: from post_util import *
# Run Time: 
# Ref: 
# ##############################################################################
# 20210524, In Kyu Lee
# Desc: Postprocessing utility
# ##############################################################################
#  - opening
#  - closing
# ##############################################################################
import numpy as np
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

def run_open(subj_path,kernel=[1,1,5]):
    pred = sitk.ReadImage(subj_path)
    img = sitk.GetArrayFromImage(pred)
    img = img/255
    img = img.astype(np.uint8)
    img = sitk.GetImageFromArray(img)
    open_img = sitk.BinaryOpeningByReconstruction(img, kernel)
    open_img = sitk.GetArrayFromImage(open_img)
    return np.transpose(open_img*255,(2,1,0))

def run_close(subj_path,kernel=[5,5,5]):
    pred = sitk.ReadImage(subj_path)
    img = sitk.GetArrayFromImage(pred)
    img = img/255
    img = img.astype(np.uint8)
    img = sitk.GetImageFromArray(img)
    close_img = sitk.BinaryClosingByReconstruction(img, kernel)
    close_img = sitk.GetArrayFromImage(close_img)
    return np.transpose(close_img*255,(2,1,0))