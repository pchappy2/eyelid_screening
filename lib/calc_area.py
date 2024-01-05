import os
import numpy as np
import matplotlib.pyplot as plt


def get_map(eye_seg_map):
    sclera_map = np.zeros(eye_seg_map.shape, dtype=np.int32)
    iris_map = np.zeros(eye_seg_map.shape, dtype=np.int32)
    sclera_map[eye_seg_map == 1] = 1
    iris_map[eye_seg_map == 2] = 1

    return sclera_map, iris_map


def split_eye_map(eye_seg_map, pupil_coord):
    left_map = eye_seg_map[:, :pupil_coord[1]]
    right_map = eye_seg_map[:, pupil_coord[1]:]

    return left_map, right_map


def calc_area(eye_map, pupil):
    left_map, right_map = split_eye_map(eye_map, pupil)
    left_sclera_map, left_iris_map = get_map(left_map)
    right_sclera_map, right_iris_map = get_map(right_map)
    left_sclera_area = np.sum(left_sclera_map)
    right_sclera_area = np.sum(right_sclera_map)
    iris_area = np.sum(left_iris_map) + np.sum(right_iris_map)

    sclera_area = left_sclera_area + right_sclera_area

    return left_sclera_area, right_sclera_area, iris_area


def trans_str2int(coord_str):
    str_y, str_x = coord_str.split(",")

    return int(str_y), int(str_x)


def get_pupil_coord(pupil_str, eye_str):
    pupil_y, pupil_x = trans_str2int(pupil_str)
    eye_y, eye_x = trans_str2int(eye_str)

    pupil_coord = (pupil_y - eye_y, pupil_x - eye_x)

    return pupil_coord


if __name__ == "__main__":
    import pandas as pd

    path = "H:/ptosis_normal_and_patient/normal"
    face_path = os.path.join(path, "origin")
    eyes_seg_path = os.path.join(path, "eyes_seg_map")
    eyes_path = os.path.join(path, "eyes")
    rotated_path = os.path.join(path, "rotated_origin")
    eyes_file = os.path.join(path, "rotated_eye_loc.csv")

    eval_df = pd.DataFrame(columns=['file', 'left_lateral_area', 'left_medial_area', 'left_iris_area',
                                    'right_lateral_area', 'right_medial_area', 'right_iris_area'])

    eyes_df = pd.read_csv(eyes_file)
    for i in range(eyes_df.shape[0]):
        f = eyes_df.loc[i, 'file']
        fname = f.split(".")[0]
        left_seg_file = fname + "_left.npy"
        right_seg_file = fname + "_right.npy"
        left_seg_eye = np.load(os.path.join(eyes_seg_path, left_seg_file))
        right_seg_eye = np.load(os.path.join(eyes_seg_path, right_seg_file))
        left_pupil = get_pupil_coord(eyes_df.loc[i, 'left_min'], eyes_df.loc[i, 'left_pupil'])
        right_pupil = get_pupil_coord(eyes_df.loc[i, 'right_min'], eyes_df.loc[i, 'right_pupil'])

        left_medial_area, left_lateral_area, left_iris_area = calc_area(left_seg_eye, left_pupil)
        right_lateral_area, right_medial_area, right_iris_area = calc_area(right_seg_eye, right_pupil)

        tmp_df = pd.DataFrame(data={'file': f,
                                    'left_lateral_area': left_lateral_area,
                                    'left_medial_area': left_medial_area,
                                    'left_iris_area':left_iris_area,
                                    'right_lateral_area': right_lateral_area,
                                    'right_medial_area': right_medial_area,
                                    'right_iris_area': right_iris_area}, index=[0])
        eval_df = eval_df.append(tmp_df, ignore_index=True, sort=False)

    eval_df.to_excel(os.path.join(path, "evaluation.xlsx"), index=False)