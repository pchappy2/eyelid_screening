import os
import numpy as np
import pandas as pd
import cv2
from skimage.measure import label


def draw_contour(eye_seg_map):
    eyelid = np.zeros(eye_seg_map.shape, dtype=np.uint8)
    eyelid[eye_seg_map > 0] = 255
    eye_contours = np.zeros((eyelid.shape[0], eyelid.shape[1], 3), dtype=np.uint8)
    ret, binary = cv2.threshold(eyelid, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(eye_contours, contours, -1, (255, 255, 255), 1)

    return eye_contours


def find_canthus(eye_contours):
    eye_contours = cv2.cvtColor(eye_contours, cv2.COLOR_BGR2GRAY)
    contour_coords = np.where(eye_contours >= 100)
    left_x, right_x = np.min(contour_coords[1]), np.max(contour_coords[1])
    left_slice = eye_contours[:, left_x]
    right_slice = eye_contours[:, right_x]

    left_y = np.where(left_slice >= 100)[0][0]
    right_y = np.where(right_slice >= 100)[0][0]

    return (left_y, left_x), (right_y, right_x)


def divide_up_and_low(eye_contours, left_canthus, right_canthus):
    eye_contours = cv2.cvtColor(eye_contours, cv2.COLOR_BGR2GRAY)
    eye_contours = eye_contours.astype(np.int32)
    eye_contours[eye_contours < 127] = 0
    eye_contours[eye_contours >= 127] = 1
    eye_contours[left_canthus[0], left_canthus[1]-5:left_canthus[1]+5] = 0
    eye_contours[right_canthus[0], right_canthus[1] - 5:right_canthus[1] + 5] = 0

    eyelid_labels = label(eye_contours)

    #print(np.where(eyelid_bound == 1))
    return eyelid_labels


def draw_up_and_low(eyelid_bound):
    eyelid_color = np.zeros(eyelid_bound.shape, dtype=np.uint8)
    eyelid_color[np.where(eyelid_bound == 1)] = 255
    eyelid_color[np.where(eyelid_bound == 2)] = 128

    return eyelid_color


def calc_lid_length(eye_seg_map):
    eye_contour = draw_contour(eye_seg_map)
    left_canthus, right_canthus = find_canthus(eye_contour)
    eyelid_bound = divide_up_and_low(eye_contour, left_canthus, right_canthus)

    lid1 = np.zeros(eyelid_bound.shape)
    lid2 = np.zeros(eyelid_bound.shape)
    lid1[np.where(eyelid_bound == 1)] = 1    # 睑缘1
    lid2[np.where(eyelid_bound == 2)] = 1    # 睑缘2

    lid_ys1 = np.where(lid1 == 1)   # 睑缘1纵坐标
    lid_ys2 = np.where(lid2 == 1)   # 睑缘2纵坐标

    if np.max(lid_ys1) > np.max(lid_ys2):   # 若睑缘1最大纵坐标 > 睑缘2, 1在下2在上
        up_lid = lid2
        low_lid = lid1
    else:
        up_lid = lid1
        low_lid = lid2

    return np.sum(up_lid), np.sum(low_lid)


if __name__ == "__main__":

    path = "H:/ptosis_normal_and_patient/patient"
    contour_path = os.path.join(path, "eyelid_mask")
    eye_seg_path = os.path.join(path, "eyes_seg_map")
    evaluation_file = os.path.join(path, "evaluation.xlsx")

    evaluation_df = pd.read_excel(evaluation_file)
    for i in range(evaluation_df.shape[0]):
        f = evaluation_df.loc[i, 'file']
        left_eye_file = os.path.join(eye_seg_path, f.split(".")[0]+"_left.npy")
        right_eye_file = os.path.join(eye_seg_path, f.split(".")[0]+"_right.npy")
        left_eye_seg = np.load(left_eye_file)
        right_eye_seg = np.load(right_eye_file)

        left_eye_contours = draw_contour(left_eye_seg)
        right_eye_contours = draw_contour(right_eye_seg)

        left_medial_canthus, left_lateral_canthus = find_canthus(left_eye_contours)
        right_lateral_canthus, right_medial_canthus = find_canthus(right_eye_contours)

        #tmp_df = pd.DataFrame(data={'file':i, 'left_canthus':str(left_canthus[0])+","+str(left_canthus[1]),
        #                           'right_canthus':str(right_canthus[0])+","+str(right_canthus[1])}, index=[0])
        #canthus_df = canthus_df.append(tmp_df, ignore_index=True, sort=False)
        left_eyelid = divide_up_and_low(left_eye_contours, left_medial_canthus, left_lateral_canthus)
        right_eyelid = divide_up_and_low(right_eye_contours, right_lateral_canthus, right_medial_canthus)
        left_lid_color = draw_up_and_low(left_eyelid)
        right_lid_color = draw_up_and_low(right_eyelid)
        cv2.imwrite(os.path.join(path, "eyelid_color", f.split(".")[0]+"_left.png"), left_lid_color)
        cv2.imwrite(os.path.join(path, "eyelid_color", f.split(".")[0]+"_right.png"), right_lid_color)
        left_up_lid_len, left_low_lid_len = calc_lid_length(left_eyelid)
        right_up_lid_len, right_low_lid_len = calc_lid_length(right_eyelid)
        evaluation_df.loc[i, 'left_upper_lid_length'] = left_up_lid_len
        evaluation_df.loc[i, 'left_lower_lid_length'] = left_low_lid_len
        evaluation_df.loc[i, 'right_upper_lid_length'] = right_up_lid_len
        evaluation_df.loc[i, 'right_lower_lid_length'] = right_low_lid_len
        #np.save(os.path.join(contour_path, i), eyelid)

    evaluation_df.to_excel(os.path.join(path, "evaluation.xlsx"), index=False)