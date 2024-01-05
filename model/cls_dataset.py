import os
import random

import cv2

import numpy as np
import pandas as pd

from torch.utils import data


class EyelidData(data.Dataset):
    def __init__(self, file, stage="train"):
        if stage == "train":
            self.flist = self.choice_list(flist, ratio=0.1)

    @staticmethod
    def choice_daojie(file, ratio=0.1):
        df = pd.read_csv(file)
        waifan_df = df[df["label"] == 2].reset_index(drop=True)
        daojie_df = df[df["label"] == 1].reset_index(drop=True)
        normal_df = df[df["label"] == 0].reset_index(drop=True)
        sample_num = int(waifan_df.shape[0] * ratio)
        daojie_ids = np.random.choice(len(daojie_df.shape[0]), sample_num, replace=True)
        daojie_new_df = pd.DataFrame(columns=["file", "label"])
        for i in range(daojie_df.shape[0]):
            if i in daojie_ids:
                tmp = pd.DataFrame

        # return img_list


class NeiziData(data.Dataset):
    def __init__(self, flist_file, trg_size=(256, 256)):
        with open(flist_file, "r") as obj:
            lines = obj.readlines()
        self.data_list = [i.rstrip() for i in lines]
        self.target_size = trg_size

    def __getitem__(self, idx):
        sample = self.data_list[idx].split(",")
        # print(sample[0])
        image = cv2.imread(sample[0])
        rs_img = cv2.resize(image, dsize=self.target_size).astype(np.float32)
        rs_img = rs_img.transpose(2, 0, 1)
        rs_img /= 255.
        target = int(sample[1])

        return rs_img, target

    @staticmethod
    def flip(img):
        axis = random.randint(-1, 1)
        if axis == -1:
            return img
        elif axis == 0:
            return img[::-1, ...]
        else:
            return img[:, ::-1, :]

    @staticmethod
    def rotate(img):
        angle = random.randint(-1, 2)
        if angle == -1:
            return img
        elif angle == 0:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 1:
            return cv2.rotate(img, cv2.ROTATE_180)
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def __len__(self):
        return len(self.data_list)


class BtEyelidData(data.Dataset):
    def __init__(self, flist_file, trg_size=(128, 256)):
        with open(flist_file, "r") as obj:
            lines = obj.readlines()
        self.data_list = [i.rstrip() for i in lines]
        self.target_size = trg_size

    def __getitem__(self, idx):
        sample = self.data_list[idx].split(",")
        # print(sample[0])
        image = cv2.imread(sample[0])
        rs_img = cv2.resize(image, dsize=self.target_size).astype(np.float32)
        rs_img = rs_img.transpose(2, 0, 1)
        rs_img /= 255.
        target = int(sample[1])

        return rs_img, target

    @staticmethod
    def flip(img):
        axis = random.randint(-1, 1)
        if axis == -1:
            return img
        elif axis == 0:
            return img[::-1, ...]
        else:
            return img[:, ::-1, :]

    @staticmethod
    def rotate(img):
        angle = random.randint(-1, 2)
        if angle == -1:
            return img
        elif angle == 0:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 1:
            return cv2.rotate(img, cv2.ROTATE_180)
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def __len__(self):
        return len(self.data_list)



