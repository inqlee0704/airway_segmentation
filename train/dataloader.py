""" ****************************************** 
    Author: In Kyu Lee
    Deep learning dataloaders are stored here.
    Available:
    - ImageDataset: classification
    - SegDataset: Semantic segmentation
    - slice_loader: load slice information for SegDataset
    - CT_loader: load CT images
    - SlicesDataset: Semantic Segmentation (load all images into memory)
    - check_files: check if ct and mask file exist
****************************************** """
import os
from medpy.io import load
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import exposure
import torch.nn.functional as F

"""
ImageDataset for 3D CT image Segmentation
Inputs:
    - subjlist: panda's dataframe which contains image & mask paths [df]
    - slices: slice information from slice_loader function [list]
Outputs:
    - dictionary that containts both image tensor & mask tensor [dict]

"""


class SegDataset:
    def __init__(
        self,
        subjlist,
        slices,
        mask_name=None,
        augmentations=None,
        DEBUG=False,
        TEST=False,
    ):
        if DEBUG:
            self.subj_paths = subjlist.loc[:10, "ImgDir"].values
        else:
            self.subj_paths = subjlist.loc[:, "ImgDir"].values

        if TEST:
            self.img_paths = self.subj_paths

        else:
            self.img_paths = [
                os.path.join(subj_path, "zunu_vida-ct.img")
                for subj_path in self.subj_paths
            ]
            if mask_name == "airway":
                self.mask_paths = [
                    os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
                    for subj_path in self.subj_paths
                ]
            elif mask_name == "lung":
                self.mask_paths = [
                    os.path.join(subj_path, "ZUNU_vida-lung.img.gz")
                    for subj_path in self.subj_paths
                ]

        self.slices = slices
        self.pat_num = None
        self.img = None
        self.mask = None
        self.mask_name = mask_name
        self.augmentations = augmentations
        self.TEST = TEST

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img, _ = load(self.img_paths[slc[0]])
            # some background values are set to -3024
            self.img[self.img < -1024] = -1024
            if not self.TEST:
                self.mask, _ = load(self.mask_paths[slc[0]])
            self.pat_num = slc[0]
        img = self.img[:, :, slc[1]]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img[None, :]

        if not self.TEST:
            mask = self.mask[:, :, slc[1]]
            # Airway mask is stored as 255
            if self.mask_name == "airway":
                mask = mask / 255
            elif self.mask_name == "lung":
                mask[mask == 20] = 1
                mask[mask == 30] = 2
            else:
                print("Specify mask_name (airway or lung)")
                return -1

            if self.augmentations is not None:
                augmented = self.augmentations(image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]

            return {
                "image": torch.tensor(img.copy(), dtype=torch.float),
                "seg": torch.tensor(mask.copy(), dtype=torch.long),
            }

        else:  # Test mode
            return {
                "image": torch.tensor(img.copy(), dtype=torch.float),
            }


class SegDataset_Z:
    def __init__(
        self,
        subjlist,
        slices,
        mask_name=None,
        augmentations=None,
        DEBUG=False,
        TEST=False,
    ):
        if DEBUG:
            self.subj_paths = subjlist.loc[:10, "ImgDir"].values
        else:
            self.subj_paths = subjlist.loc[:, "ImgDir"].values

        if TEST:
            self.img_paths = self.subj_paths

        else:
            self.img_paths = [
                os.path.join(subj_path, "zunu_vida-ct.img")
                for subj_path in self.subj_paths
            ]
            if mask_name == "airway":
                self.mask_paths = [
                    os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
                    for subj_path in self.subj_paths
                ]
            elif mask_name == "lung":
                self.mask_paths = [
                    os.path.join(subj_path, "ZUNU_vida-lung.img.gz")
                    for subj_path in self.subj_paths
                ]

        self.slices = slices
        self.pat_num = None
        self.img = None
        self.mask = None
        self.mask_name = mask_name
        self.augmentations = augmentations
        self.TEST = TEST

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img, _ = load(self.img_paths[slc[0]])
            # some background values are set to -3024
            self.img[self.img < -1024] = -1024
            if not self.TEST:
                self.mask, _ = load(self.mask_paths[slc[0]])
            self.pat_num = slc[0]
        img = self.img[:, :, slc[1]]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = img[None, :]

        z = slc[1] / (self.img.shape[2] + 1)
        # z ranges from 0 to 9
        z = np.floor(z * 10)

        if not self.TEST:
            mask = self.mask[:, :, slc[1]]
            # Airway mask is stored as 255
            if self.mask_name == "airway":
                mask = mask / 255
            elif self.mask_name == "lung":
                mask[mask == 20] = 1
                mask[mask == 30] = 2
            else:
                print("Specify mask_name (airway or lung)")
                return -1

            if self.augmentations is not None:
                augmented = self.augmentations(image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]

            return {
                "image": torch.tensor(img.copy(), dtype=torch.float),
                "seg": torch.tensor(mask.copy(), dtype=torch.long),
                "z": torch.tensor(z, dtype=torch.int64),
            }

        else:  # Test mode
            return {
                "image": torch.tensor(img.copy(), dtype=torch.float),
                "z": torch.tensor(z, dtype=torch.int64),
            }


class SegDataset_multiC_withZ:
    def __init__(
        self, subjlist, slices, mask_name=None, resize=None, augmentations=None
    ):
        self.subj_paths = subjlist.loc[:, "ImgDir"].values
        self.img_paths = [
            os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in self.subj_paths
        ]
        if mask_name == "airway":
            self.mask_paths = [
                os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
                for subj_path in self.subj_paths
            ]
        elif mask_name == "lung":
            self.mask_paths = [
                os.path.join(subj_path, "ZUNU_vida-lung.img.gz")
                for subj_path in self.subj_paths
            ]
        self.slices = slices
        self.pat_num = None
        self.img = None
        self.hdr = None
        self.mask = None
        self.mask_name = mask_name
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img, self.hdr = load(self.img_paths[slc[0]])
            # some background values are set to -3024
            self.img[self.img < -1024] = -1024
            self.narrow_c = np.copy(self.img)
            self.wide_c = np.copy(self.img)
            self.narrow_c[self.narrow_c >= -500] = -500
            # self.wide_c[self.wide_c <= -200] = -200
            self.wide_c[self.wide_c >= 300] = 300

            self.img = (self.img - np.min(self.img)) / (
                np.max(self.img) - np.min(self.img)
            )
            self.wide_c = (self.wide_c - np.min(self.wide_c)) / (
                np.max(self.wide_c) - np.min(self.wide_c)
            )
            self.narrow_c = (self.narrow_c - np.min(self.narrow_c)) / (
                np.max(self.narrow_c) - np.min(self.narrow_c)
            )

            self.mask, _ = load(self.mask_paths[slc[0]])
            if self.mask_name == "airway":
                self.mask = self.mask / 255
            elif self.mask_name == "lung":
                self.mask[self.mask == 20] = 1
                self.mask[self.mask == 30] = 1
            else:
                print("Specify mask_name (airway or lung)")
                return -1

            self.pat_num = slc[0]

        z = slc[1] / (self.img.shape[2] + 1)
        # z ranges from 0 to 9
        z = np.floor(z * 10)

        narrow_c = self.narrow_c[:, :, slc[1]]
        wide_c = self.wide_c[:, :, slc[1]]
        img = self.img[:, :, slc[1]]
        mask = self.mask[:, :, slc[1]]
        mask = mask.astype(int)

        narrow_c = narrow_c[None, :]
        wide_c = wide_c[None, :]
        img = img[None, :]
        mask = mask[None, :]
        img = np.concatenate([img, narrow_c, wide_c], axis=0)

        if self.augmentations is not None:
            augmented = self.augmentations(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        return {
            "image": torch.tensor(img.copy()),
            "seg": torch.tensor(mask.copy()),
            "z": torch.tensor(z, dtype=torch.int64),
        }



"""
Slice loader which outputs slice information for each CT
and check if mask dimension and image dimension match
Inputs:
    - subjlist: panda's dataframe which contains image & mask paths [df]
Outputs:
    - A slice list of tuples, [list]
        - first index represent subject's number
        - second index represent axial position of CT
    ex) (0,0),(0,1),(0,2) ... (0,750),(1,0),(1,1) ... (300, 650)
"""


def slice_loader(subjlist, TEST=False):
    print("Loading Data")
    subj_paths = subjlist.loc[:, "ImgDir"].values

    slices = []
    if TEST:
        img_paths = subj_paths
        for ii in range(len(img_paths)):
            img, _ = load(img_paths[ii])
            for jj in range(img.shape[2]):
                slices.append((ii, jj))
    else:
        img_paths = [
            os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in subj_paths
        ]
        mask_paths = [
            os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
            for subj_path in subj_paths
        ]

        for ii in range(len(mask_paths)):
            label, _ = load(mask_paths[ii])
            img, _ = load(img_paths[ii])
            if img.shape != label.shape:
                print("Dimension does not match: ")
                print(subjlist.loc[ii, "ImgDir"])
            for jj in range(label.shape[2]):
                slices.append((ii, jj))
    return slices


def TE_loader(subjlist, multi_c=False):
    print("Loading Data. . . ")
    dicom_paths = subjlist.loc[:, "ImgDir"].values
    out = []
    if multi_c:
        for ii in range(len(dicom_paths)):
            raw_image, _ = load(dicom_paths[ii])
            raw_image[raw_image < -1024] = -1024
            image = np.copy(raw_image)
            narrow_c = np.copy(raw_image)
            wide_c = np.copy(raw_image)
            narrow_c[narrow_c >= -500] = -500
            wide_c[wide_c >= 300] = 300
            image = (image - (np.min(image))) / ((np.max(image) - (np.min(image))))
            narrow_c = (narrow_c - (np.min(narrow_c))) / (
                (np.max(narrow_c) - (np.min(narrow_c)))
            )
            wide_c = (wide_c - (np.min(wide_c))) / ((np.max(wide_c) - (np.min(wide_c))))
            image = image[None, :]
            narrow_c = narrow_c[None, :]
            wide_c = wide_c[None, :]
            img_combined = np.concatenate([image, narrow_c, wide_c], axis=0)
            out.append({"image": img_combined})
        return np.array(out)
    else:
        for ii in range(len(dicom_paths)):
            raw_image, _ = load(dicom_paths[ii])
            raw_image[raw_image < -1024] = -1024
            image = np.copy(raw_image)
            image = (image - (np.min(image))) / ((np.max(image) - (np.min(image))))
            out.append({"image": image})
        return np.array(out)


"""
Check files before the train
"""


def check_files(subjlist):
    subj_paths = subjlist.loc[:, "ImgDir"].values
    img_path_ = [
        os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in subj_paths
    ]
    mask_path_ = [
        os.path.join(subj_path, "ZUNU_vida-airtree.img.gz") for subj_path in subj_paths
    ]
    for i in range(len(img_path_)):
        if not os.path.exists(img_path_[i]):
            print(img_path_[i], "Not exists")
        if not os.path.exists(mask_path_[i]):
            print(mask_path_[i], "Not exists")


"""
Prepare train & valid dataloaders
"""


def prep_dataloader(c, k=None, df=None):
    # n_case: load n number of cases, 0: load all
    # K is not none, implement KFold
    if k is not None:
        df_train = df[df['fold']!=k].reset_index(drop=True)
        df_valid = df[df['fold']==k].reset_index(drop=True)
    else:
        df_train = pd.read_csv(os.path.join(c.data_path, c.in_file), sep="\t")
        df_valid = pd.read_csv(os.path.join(c.data_path, c.in_file_valid), sep="\t")

    if c.debug:
        df_train = df_train[:1]
        df_valid = df_train[:1]

    train_slices = slice_loader(df_train)
    valid_slices = slice_loader(df_valid)

    aug = get_train_aug()
    if c.Z:
        train_ds = SegDataset_Z(
            df_train, train_slices, mask_name=c.mask, augmentations=aug
        )
        valid_ds = SegDataset_Z(df_valid, valid_slices, mask_name=c.mask)

    else:
        train_ds = SegDataset(
            df_train,
            train_slices,
            mask_name=c.mask,
            augmentations=aug,
        )
        valid_ds = SegDataset(df_valid, valid_slices, mask_name=c.mask)

    train_loader = DataLoader(
        train_ds, batch_size=c.train_bs, shuffle=False, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=c.valid_bs, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader


def prep_testloader(infer_path, test_bs=1):
    # n_case: load n number of cases, 0: load all
    df = pd.read_csv(infer_path)
    slices = slice_loader(df, TEST=True)
    ds = SegDataset(
        df,
        slices,
        TEST=True,
    )
    dataloader = DataLoader(ds, batch_size=test_bs, shuffle=False, num_workers=0)
    return dataloader


def prep_dataloader_z(c):
    # n_case: load n number of cases, 0: load all
    df_train = pd.read_csv(os.path.join(c.data_path, c.in_file), sep="\t")
    df_valid = pd.read_csv(os.path.join(c.data_path, c.in_file_valid), sep="\t")

    train_slices = slice_loader(df_train)
    valid_slices = slice_loader(df_valid)

    train_loader = DataLoader(
        train_ds, batch_size=c.train_bs, shuffle=False, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=c.valid_bs, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader


def prep_dataloader_multiC_z(c):
    # n_case: load n number of cases, 0: load all
    df_train = pd.read_csv(os.path.join(c.data_path, c.in_file), sep="\t")
    df_valid = pd.read_csv(os.path.join(c.data_path, c.in_file_valid), sep="\t")

    train_slices = slice_loader(df_train)
    valid_slices = slice_loader(df_valid)

    train_ds = SegDataset_multiC_withZ(
        df_train, train_slices, mask_name=c.mask, augmentations=get_train_aug()
    )
    valid_ds = SegDataset_multiC_withZ(df_valid, valid_slices, mask_name=c.mask)

    train_loader = DataLoader(
        train_ds, batch_size=c.train_bs, shuffle=False, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=c.valid_bs, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader


def get_train_aug():
    return A.Compose(
        [
            A.Rotate(limit=15),
            A.OneOf(
                [
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=5),
                    A.MotionBlur(blur_limit=7),
                    A.GaussianBlur(blur_limit=(3, 7)),
                ],
                p=0.5,
            ),
            # ToTensorV2()
        ]
    )


# def get_valid_aug():
#     return A.Compose([
#         A.OneOf([
#             A.HorizontalFlip(),
#             A.VerticalFlip(),
#         ],p=0.4),
#     ])


def prep_test_img(multiC=False):
    # Test
    test_img, _ = load("/data1/inqlee0704/silicosis/data/inputs/02_ct.hdr")
    test_img[test_img < -1024] = -1024
    if multiC:
        narrow_c = np.copy(test_img)
        wide_c = np.copy(test_img)
        narrow_c[narrow_c >= -500] = -500
        wide_c[wide_c >= 300] = 300
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
        wide_c = (wide_c - np.min(wide_c)) / (np.max(wide_c) - np.min(wide_c))
        narrow_c = (narrow_c - np.min(narrow_c)) / (np.max(narrow_c) - np.min(narrow_c))
        narrow_c = narrow_c[None, :]
        wide_c = wide_c[None, :]
        test_img = test_img[None, :]
        test_img = np.concatenate([test_img, narrow_c, wide_c], axis=0)
    else:
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
        # test_img = test_img[None, :]
    return test_img
