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
import numpy as np
from medpy.io import load,save
import SimpleITK as sitk

sitk.PorcessObject_SetGlobalWarningDisplay(False)

# load images