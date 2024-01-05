import os
import traceback

import cv2
import torch
import json
import numpy as np
from model.atten_unet import AttU_Net
from collections import OrderedDict


def load_eyenet(ckpt):
    """
    load model
    :type ckpt: str
    """
    seg_net = AttU_Net(img_ch=3, output_ch=3)
    seg_net = seg_net.cuda()
    state_dict = torch.load(ckpt)['net']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    seg_net.load_state_dict(new_state_dict)
    seg_net.eval()

    return seg_net


def read_eye_coords_txt(eye_txt_file):
    eye_coords = []
    with open(eye_txt_file, "r") as obj:
        lines = obj.readlines()
    for line in lines:
        eye_coords.append([int(i) for i in line.rstrip().split(" ")])

    return eye_coords


def get_eye_patch(face_img, eye_coords):
    eyes = []
    patch_coords = []
    for eye_c in eye_coords:
        xmin, ymin, xmax, ymax = eye_c[0], eye_c[1], eye_c[2], eye_c[3]
        xmin = max(xmin - 15, 0)
        ymin = max(ymin - 15, 0)
        xmax = min(face_img.shape[1], xmax + 15)
        ymax = min(face_img.shape[0], ymax + 15)
        eye_patch = face_img[ymin:ymax+1, xmin:xmax+1, :]
        eyes.append(eye_patch)
        patch_coords.append((xmin, ymin, xmax, ymax))

    return eyes, patch_coords


def img_preprocess(img, size=(512, 256)):
    height, width, _ = img.shape
    img = cv2.resize(img, size)
    img = img.astype(np.float32)
    img /= 255.
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)

    return img, (width, height)


def seg(net, img, size=(512, 256)):
    img, origin_size = img_preprocess(img, size)
    input = torch.FloatTensor(img).cuda()
    output = net(input)
    _, pred = torch.topk(output, k=1, dim=1)
    pred = pred.squeeze()
    pred = pred.data.cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, origin_size)
    pred = pred.astype(np.int32)

    return pred


def build_face_seg(face_img, eye_segs, eye_coords):
    face_seg = np.zeros(face_img.shape, dtype=np.uint8)
    face_msk = np.zeros((face_img.shape[0], face_img.shape[1]), dtype=np.uint8)

    for i in range(len(eye_segs)):
        eye_pred = eye_segs[i]
        eye_coord = eye_coords[i]  # xyxy
        # eye_seg = np.zeros((eye_pred.shape[0], eye_pred.shape[1], 3), dtype=np.uint8)
        face_msk[eye_coord[1]:eye_coord[3]+1, eye_coord[0]:eye_coord[2]+1] = eye_pred

    face_seg[face_msk == 1] = (0, 255, 0)
    face_seg[face_msk == 2] = (0, 0, 255)
    face_overlap = cv2.addWeighted(face_img, 0.6, face_seg, 0.4, gamma=0)

    return face_seg, face_overlap


if __name__ == "__main__":
    eye_net = load_eyenet("./results/eye.pth")
    root_path = "/data/feituai/eyelid_screen/"
    face_img_path = root_path + "origin/"
    eye_coord_path = root_path + "predict_txt_220604/"
    save_path = root_path + "face_overlap/"
    npy_path = root_path + "face_seg/"

    # sub_dir = os.listdir(face_img_path)
    sub_dir = ["re_trichiasis"]
    for sub in sub_dir:
        sub_img_path = face_img_path + sub + "/"
        sub_eye_coord_path = eye_coord_path + sub + "/"
        sub_save_path = save_path + sub + "/"
        sub_npy_path = npy_path + sub + "/"
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        if not os.path.exists(sub_npy_path):
            os.makedirs(sub_npy_path)
        face_img_list = os.listdir(sub_img_path)
        for file in face_img_list:
            img_file = sub_img_path + file
            fname = os.path.splitext(file)[0]
            coord_js_file = sub_eye_coord_path + fname + ".json"
            if os.path.exists(coord_js_file):
                continue
            else:
                coord_file = sub_eye_coord_path + fname + ".txt"
            if os.path.exists(coord_file):
                try:
                    eye_coords = read_eye_coords_txt(coord_file)
                    face_img = cv2.imread(img_file)
                    eye_patchs, coord_patchs = get_eye_patch(face_img, eye_coords)
                    eye_segs = []
                    for eye_patch in eye_patchs:
                        eye_pred = seg(eye_net, eye_patch)
                        eye_segs.append(eye_pred)
                    face_seg, face_result = build_face_seg(face_img, eye_segs, coord_patchs)
                    np.save(sub_npy_path + fname + ".npy", face_seg)
                    save_file = sub_save_path + file
                    cv2.imwrite(save_file, face_result)
                except Exception:
                    print(img_file)
                    print(traceback.format_exc())
