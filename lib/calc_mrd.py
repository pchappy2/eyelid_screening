import os
import pandas as pd
import numpy as np
from lib.detect_pupil import detect_pupil2, get_contour


def calc_mrd(eye_seg_map, pupil_coord):
    seg_slice = eye_seg_map[:, pupil_coord[1]]
    coords = np.where(seg_slice > 0)[0]
    min_y = np.min(coords)
    max_y = np.max(coords)
    mrd1 = pupil_coord[0] - min_y
    mrd2 = max_y - pupil_coord[0]
    pf = mrd1 + mrd2

    return mrd1, mrd2, pf


if __name__ == "__main__":

    path = "H:/ptosis_normal_and_patient/normal"
    contour_path = os.path.join(path, "eyelid_mask")
    eye_seg_path = os.path.join(path, "eyes_seg_map")
    evaluation_file = os.path.join(path, "evaluation.xlsx")
    eyes_file = os.path.join(path, "rotated_eye_loc.csv")
    eyes_df = pd.read_csv(eyes_file)

    evaluation_df = pd.read_excel(evaluation_file)
    for i in range(evaluation_df.shape[0]):
        f = evaluation_df.loc[i, 'file']
        print(f)
        eye_tmp_df = eyes_df[eyes_df['file'] == f]
        #left_pupil = eye_tmp_df.iloc[0, 4]
        #right_pupil = eye_tmp_df.iloc[0, 3]
        #left_eye_coord = eye_tmp_df.iloc[0, 2]
        #right_eye_coord = eye_tmp_df.iloc[0, 1]
        #left_pupil_coord = get_pupil_coord(left_pupil, left_eye_coord)
        #right_pupil_coord = get_pupil_coord(right_pupil, right_eye_coord)

        left_eye_file = os.path.join(eye_seg_path, f.split(".")[0]+"_left.npy")
        right_eye_file = os.path.join(eye_seg_path, f.split(".")[0]+"_right.npy")
        left_eye_seg = np.load(left_eye_file)
        right_eye_seg = np.load(right_eye_file)
        left_iris_seg, _ = get_contour(left_eye_seg)
        right_iris_seg, _ = get_contour(right_eye_seg)
        left_pupil_coord = detect_pupil2(left_iris_seg)
        right_pupil_coord = detect_pupil2(right_iris_seg)
        left_mrd1, left_mrd2, left_pf = calc_mrd(left_eye_seg, left_pupil_coord)
        right_mrd1, right_mrd2, right_pf = calc_mrd(right_eye_seg, right_pupil_coord)
        print(left_mrd1, left_mrd2, left_pf)
        print(right_mrd1, right_mrd2, right_pf)

        evaluation_df.loc[i, 'left_MDR1'] = left_mrd1
        evaluation_df.loc[i, 'left_MDR2'] = left_mrd2
        evaluation_df.loc[i, 'left_PF'] = left_pf
        evaluation_df.loc[i, 'right_MDR1'] = right_mrd1
        evaluation_df.loc[i, 'right_MDR2'] = right_mrd2
        evaluation_df.loc[i, 'right_PF'] = right_pf

    evaluation_df.to_excel(evaluation_file, index=False)