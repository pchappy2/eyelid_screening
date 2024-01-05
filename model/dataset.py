import os
import random

import cv2
import torch
import torch.nn.functional as F
import random
import torch.utils.data as data
from scipy.ndimage.interpolation import zoom

import numpy as np


class TumorData(data.Dataset):
    def __init__(self, flist_file, size=(512, 512)):
        with open(flist_file, "r") as file:
            self.flist = file.readlines()
        self.size = size

    def __getitem__(self, idx):
        arrs = np.load(self.flist[idx].rstrip())
        img = arrs["img"]
        msk = arrs["msk"].astype(np.uint8)
        img, msk = self.process(img=img, msk=msk)
        img, msk = self.random_flip(img, msk)
        img = img.transpose(2, 0, 1)
        # img = np.expand_dims(img, 0)
        msk = np.expand_dims(msk, 0)
        img = img.astype(np.float32)
        msk = msk.astype(np.float32)
        img /= 255.

        return img.copy(), msk.copy()

    def process(self, img, msk):
        height, width, _ = img.shape
        rs_img = cv2.resize(src=img.copy(), dsize=self.size)
        rs_msk = cv2.resize(src=msk.copy(), dsize=self.size)

        return rs_img, rs_msk

    @staticmethod
    def random_flip(img, msk):
        axis = random.randint(-1, 1)
        if axis == 0:
            r_img = img[:, ::-1, :]
            r_msk = msk[::-1, :]
        elif axis == 1:
            r_img = img[:, :, ::-1]
            r_msk = msk[:, ::-1]
        else:
            return img.copy(), msk.copy()

        return r_img, r_msk

    def __len__(self):
        return len(self.flist)

