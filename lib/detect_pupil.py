import os
import traceback

import cv2
import numpy as np
from skimage import morphology

from sklearn.cluster import MeanShift, estimate_bandwidth


def detect_pupil2(img):
    i = 1
    flag = 1
    A = (0, 0)
    B = (0, 0)
    C = (0, 0)

    while (flag):
        if np.where(img[:, i] == 255)[0].shape[0] > 0:
            arr_1 = np.where(img[:, i] == 255)[0].tolist()
            arr_2 = np.where(img[:, i + 5] == 255)[0].tolist()
            A = (arr_1[0], i)
            B = (arr_2[-1], i + 5)
            flag = 0
        else:
            i += 1

    flag = 1
    i = img.shape[1]-10
    while (flag):
        if len(np.where(img[:, i] == 255)[0].tolist()) > 0:
            arr_3 = np.where(img[:, i] == 255)[0].tolist()
            C = (arr_1[-1], i)
            flag = 0
        else:
            i -= 1

    x1, y1 = A
    x2, y2 = B
    x3, y3 = C

    t1 = (x1 - x2) / (y1 - y2)
    t2 = (x3 - x2) / (y3 - y2)
    x0 = int((y3 - y1 - t1 * (x1 + x2) + t2 * (x3 + x2)) / 2 / (t2 - t1))
    y0 = int((t1 * t2 * (x3 - x1) - t2 * (y1 + y2) + t1 * (y3 + y2)) / 2 / (t1 - t2))

    #cv.circle(img, (y0, x0), 10, 64, -1)
    #return img
    return (x0,y0)


def compute_pupil_coords(ctr_coords):

    def random_select():
        y_coords, x_coords = ctr_coords
        num = y_coords.shape[0]
        ids = np.random.choice(num, 3, replace=False)
        ys = [y_coords[i] for i in ids]
        xs = [x_coords[i] for i in ids]
        if xs[0] == xs[1] or xs[0] == xs[2] or xs[1] == xs[2]:
            return random_select()

        return ys, xs

    ys, xs = random_select()
    y1, x1 = ys[0], xs[0]
    y2, x2 = ys[1], xs[1]
    y3, x3 = ys[2], xs[2]

    t1 = (y1 - y2) / (x1 - x2)
    t2 = (y3 - y2) / (x3 - x2)

    y0 = int((x3 - x1 - t1 * (y1 + y2) + t2 * (y3 + y2)) / 2 / (t2 - t1 + 1e-5))
    x0 = int((t1 * t2 * (y3 - y1) - t2 * (x1 + x2) + t1 * (x3 + x2)) / 2 / (t1 - t2 + 1e-5))

    return y0, x0


def get_contour(eye_seg):
    iris_seg = np.zeros(eye_seg.shape, dtype=np.uint8)
    iris_seg[eye_seg == 2] = 255
    contours, hierarchy = cv2.findContours(iris_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_slice = np.zeros(iris_seg.shape, dtype=np.int32)
    cv2.drawContours(contour_slice, contours, -1, 255, 1)

    return iris_seg, contour_slice


def locate_iris_bound(eye_seg_map):
    eye_struct = np.zeros(eye_seg_map.shape, dtype=np.uint8)
    eye_struct[eye_seg_map > 0] = 255
    kernel = morphology.disk(10)
    eye_struct = morphology.erosion(eye_struct, kernel)
    new_map = eye_seg_map.copy()
    new_map[eye_struct == 0] = 0

    def extract_ctr(new_map, mode=0):
        map = np.zeros(new_map.shape, dtype=np.uint8)
        map[new_map > mode] = 255
        map_ctr, _ = cv2.findContours(map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ctr_msk = np.zeros(new_map.shape, dtype=np.uint8)
        cv2.drawContours(ctr_msk, map_ctr, -1, 255, 1)

        return ctr_msk

    eyelid_msk = extract_ctr(new_map, mode=0)
    iris_msk = extract_ctr(new_map, mode=1)
    eyelid_ctr = np.where(eyelid_msk == 255)
    iris_ctr = np.where(iris_msk == 255)
    mask = np.zeros(eye_seg_map.shape, dtype=np.uint8)
    mask[iris_ctr] = 255
    mask[eyelid_ctr] = 0
    iris_ctr = np.where(mask == 255)

    return iris_ctr


def detect_pupil(img_seg):
    iris_ctr_coords = locate_iris_bound(img_seg)
    pupils = []
    # print(iris_ctr_coords[0].shape[0])
    for i in range(500):
        try:
            pupil_coord = compute_pupil_coords(iris_ctr_coords)
        except:
            continue
        pupils.append(pupil_coord)
    pupils = np.array(pupils)
    bandwidth = estimate_bandwidth(pupils)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pupils)

    def find_center(labels, clusters):
        label_unique, counts = np.unique(labels, return_counts=True)
        label = label_unique[np.argmax(counts)]
        cluster = clusters[label]

        return cluster

    pupil = find_center(ms.labels_, ms.cluster_centers_)

    return int(pupil[0]), int(pupil[1])


def rotate(img,A,B):
    x1,y1=A
    x2,y2=B
    if len(img.shape) == 3:
        rows, cols, _ = img.shape
    elif len(img.shape) == 2:
        rows, cols = img.shape
    theta = np.arctan((x1 - x2)/(y1 - y2))/2/np.pi*360

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst, theta


def top_point(img):
    i=0
    flag=1
    while(flag):
        length=len(np.where(img[i, :] == 255)[0].tolist())
        if length>0:
            arr=np.where(img[i,:]==255)[0].tolist()
            y=arr[length//2]
            flag=0
        else:
            i+=1
    return (i,y)


def distance(A,B):
    x1,y1=A
    x2,y2=B
    d=np.power(np.power(x1-x2,2)+np.power(y1-y2,2),0.5)
    return d


if __name__ == "__main__":
    import pandas as pd
    """
    path = "E:/BaiduNetdiskDownload/ptosis_normal_and_patient/normal"
    face_path = os.path.join(path, "origin")
    eyes_seg_path = os.path.join(path, "eyes_seg_map")
    eyes_path = os.path.join(path, "eyes")
    face_save_path = os.path.join(path, "key_points")
    rotated_path = os.path.join(path, "rotated_origin")
    eyes_file = os.path.join(path, "eye_loc.csv")

    eyes_df = pd.read_csv(eyes_file)
    eyes_df['right_pupil'] = None
    eyes_df['left_pupil'] = None
    eyes_df['angle'] = 0
    face_list = os.listdir(face_path)
    for i in range(eyes_df.shape[0]):
        f = eyes_df.loc[i, 'file']
        fname = f.split(".")[0]

        face_file = os.path.join(face_path, f)
        face_img = cv2.imread(face_file)
        face_df = eyes_df[eyes_df['file'] == face_file]

        right_eye_coord_str = eyes_df.loc[i, 'right_min']
        left_eye_coord_str = eyes_df.loc[i, 'left_min']
        right_eye_coord = right_eye_coord_str.split(",")
        left_eye_coord = left_eye_coord_str.split(",")

        left_file = os.path.join(eyes_path, fname+"_left.png")
        right_file = os.path.join(eyes_path, fname+"_right.png")
        left_seg_file = os.path.join(eyes_seg_path, fname+"_left.npy")
        right_seg_file = os.path.join(eyes_seg_path, fname+"_right.npy")

        left_pupil_coord = find_pupil_coord(left_file, left_seg_file, left_eye_coord)
        right_pupil_coord = find_pupil_coord(right_file, right_seg_file, right_eye_coord)
        #cv2.circle(face_img, (left_pupil_coord[1], left_pupil_coord[0]), 4, (0, 255, 0), 4)
        #cv2.circle(face_img, (right_pupil_coord[1], right_pupil_coord[0]), 4, (0, 255, 0), 4)
        rotate_face, angle = rotate(face_img, right_pupil_coord, left_pupil_coord)
        #cv2.imwrite(os.path.join(rotated_path, f), rotate_face)
        #cv2.imwrite(os.path.join(face_save_path, f), face_img)

        left_pupil_str = str(left_pupil_coord[0]) + "," + str(left_pupil_coord[1])
        right_pupil_str = str(right_pupil_coord[0]) + "," + str(right_pupil_coord[1])
        eyes_df.loc[i, 'left_pupil'] = left_pupil_str
        eyes_df.loc[i, 'right_pupil'] = right_pupil_str
        #eyes_df.loc[i, 'angle'] = angle

    #eyes_df.to_csv(os.path.join(path, "eye_loc.csv"), index=False)
    """
    import matplotlib.pyplot as plt

    path = "E:/BaiduNetdiskDownload/ptosis_normal_and_patient/normal"
    wrong_path = os.path.join(path, "error")
    seg_path = os.path.join(path, "eyes_seg_map")

    flist = os.listdir(wrong_path)
    f = flist[0]
    if 1:
        fname = f.split(".")[0]
        left_seg_f = fname + "_left.npy"
        right_seg_f = fname + "_right.npy"
        left_seg = np.load(os.path.join(seg_path, left_seg_f))
        right_seg = np.load(os.path.join(seg_path, right_seg_f))
        left_ctr = locate_iris_bound(left_seg)

