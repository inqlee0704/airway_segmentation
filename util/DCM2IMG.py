# ##############################################################################
# from DCM2IMG import DCMtoVidaCT
# DCMtoVidaCT('E:\\VIDA\\VIDAvision2.2\\8','E:\\VIDA_more\\UNet_Seg\\8')
# ##############################################################################
# To generate analyze "zunu_vida-ct.hdr" & "zunu_vida-ct.img" from dicom images
# ##############################################################################
# Using SimpleITK
# ##############################################################################
# 02/12/2021, In Kyu Lee
#  - saveImage argument is added to save in a different path
# 01/29/2021, In Kyu Lee
#  - converted to a function such that it can be used in python
# 10/25/2020, Jiwoong Choi
#  - for each image directory
# 10/2/2020, Jiwoong Choi
#  - started using:
#    pathProj = str(sys.argv[1])
# 6/5/2019, Jiwoong Choi
#  - cleaned up.
#  - prepare Vida case folders, e.g. 1234/, in one folder.
# 9/27/2016, Jiwoong Choi
#  - generate_zunu_vida-ct_UT_2_CBNCT_20160927.py < generate_zunu_vida-ct_UT_2.py
# -4/3/2016, Babak Haghighi
#  - generate_zunu_vida-ct_UT_2.py
# ##############################################################################
import os 
import SimpleITK as sitk
import sys

def DCMtoVidaCT(pathImage,saveImage=None):

    if saveImage == None:
        print(f'Save path is  not given, image will be saved in: ')
        print(pathImage)
        saveImage = pathImage

    os.chdir(pathImage)
    path = os.getcwd()
    print("")
    print("---------------------------------------------------")
    print(" PROGRAM BEGINS (Generate zunu_vida-ct.img & .hdr)")
    print("---------------------------------------------------")
    print(" Reading project directory:", path)
    n = 0
    print("===================================================")

    nImage = 1

    i = 0
    for i in range(nImage):
        # pathDicom = pathImage + "/dicom"
        pathDicom = pathImage
        reader = sitk.ImageSeriesReader()
        filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
        reader.SetFileNames(filenamesDICOM)
        imgOriginal = reader.Execute()
        print("    The origin after creating DICOM:", imgOriginal.GetOrigin())
        # Flip the image. 
        # The files from Apollo have differnt z direction. 
        # Thus, we need to flip the image to make it consistent with Apollo.
        flipAxes = [ False, False, True ]
        flipped = sitk.Flip(imgOriginal,flipAxes,flipAboutOrigin=True)
        print("    The origin after flipping DICOM:", flipped.GetOrigin())
        # Move the origin to (0,0,0)
        # After converting dicom to .hdr with itkv4, the origin of images changes. 
        # Thus we need to reset it to (0,0,0) to make it consistent with Apollo files.
        origin = [0.0,0.0,0.0]
        flipped.SetOrigin(origin)
        print("    The origin after flipping and changing origin to [0.,0.,0.]:", flipped.GetOrigin())
        sitk.WriteImage(flipped,saveImage + "/" + "zunu_vida-ct.hdr")
        
        print("    " + "/"  + "zunu_vida-ct.img & .hdr", "----> written" )
        print("===================================================")

    print("zunu_vida-ct.img/.hdr are created for {0} images".format(n) )
    print("-------------------------------------------------")
    print(" PROGRAM ENDS (Generate zunu_vida-ct.img & .hdr)")
    print("-------------------------------------------------")
