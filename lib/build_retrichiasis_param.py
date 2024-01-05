import os
import cv2
import json
import numpy as np
import pandas as pd

from utils import fill, calc_mrd, compute_dist_and_grad, get_long_diameter_idx
from skimage import measure
from detect_pupil import detect_pupil, rotate
from calc_mrd import calc_mrd
from calc_area import calc_area
from find_eyelid_contour import calc_lid_length


def trans_map2color(pred):
    msk = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    msk[pred == 1] = (0, 255, 0)
    msk[pred == 2] = (0, 0, 255)

    return msk


class ReTrichData(object):
    def __init__(self, face_file, eyelid_file, iris_file, manual_json):
        self.face_img = cv2.imread(face_file)
        self.center_x = self.face_img.shape[2] / 2
        eyelid_img = cv2.imread(eyelid_file)
        iris_img = cv2.imread(iris_file)
        self.eye_map, self.eyelid, self.iris = self.get_eye_map(eyelid_img, iris_img)
        self.left_box, self.right_box = self.divide_left_and_right()
        self.left_eye = self.eye_map[self.left_box[0]:self.left_box[2]+1,
                                     self.left_box[1]:self.left_box[3]+1]
        self.right_eye = self.eye_map[self.right_box[0]:self.right_box[2]+1,
                                      self.right_box[1]:self.right_box[3]+1]

        self.right_pupil, self.left_pupil = self.parseJson(manual_json)

    def parseJson(self, manual_json):
        with open(manual_json, "r") as obj:
            js_data = json.load(obj)
        orgs = js_data["shapes"]
        ps = []

        for org in orgs:
            label = org["label"]
            points = org["points"]
            if label == "p":
                ps.append(points)
        right_pupil, left_pupil = self.getOrgdict(ps)

        return right_pupil, left_pupil

    def getOrgdict(self, orgs):
        print(orgs)
        coords = []
        for i in range(len(orgs)):
            o = orgs[i]
            points = np.array(o)
            xs = points[:, 0]
            xmin = xs.min()
            xmax = xs.max()
            x_center = (xmax + xmin) / 2
            coords.append(x_center)
        if len(coords) == 2:
            if self.judgeLR(coords):
                right_pupil = orgs[0][0]
                left_pupil = orgs[1][0]
            else:
                right_pupil = orgs[1][0]
                left_pupil = orgs[0][0]
        else:
            if self.judgeLR(coords):
                right_pupil = orgs[0][0]
                left_pupil = None
            else:
                right_pupil = None
                left_pupil = orgs[0][0]

        return [int(right_pupil[1]), int(right_pupil[0])], [int(left_pupil[1]), int(left_pupil[0])]

    def judgeLR(self, coords):
        assert isinstance(coords, list)
        if len(coords) == 2:
            x1, x2 = coords[0], coords[1]
            if x1 < x2:
                return 1  # 第一个为右眼（左边），第二个为左眼（右边）
            else:
                return 0  # 第一个为左眼（右边），第二个为右眼（左边）
        else:
            x1 = coords[0]
            if x1 < self.center_x:
                return 1  # 第一个为右眼（左边）
            else:
                return 0  # 第一个为左眼（右边）

    def get_eye_map(self, eyelid_img, iris_img):
        eyelid_msk = self.extract_blue(eyelid_img)
        iris_msk = self.extract_blue(iris_img)
        eye_map = eyelid_msk.copy()
        eye_map[iris_msk == 1] = 2

        return eye_map, eyelid_msk, iris_msk

    def extract_blue(self, img):
        blue = img[..., 0]
        green = img[..., 1]
        red = img[..., 2]

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask[(blue >= 190) & (green <= 40) & (red <= 40)] = 1
        mask = self.fill(mask)
        mask[mask == 255] = 1

        return mask.astype(np.int32)

    @staticmethod
    def fill(im_in):
        im_floodfill = im_in.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_in.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = im_in | im_floodfill_inv

        return im_out

    def divide_left_and_right(self):
        label_map = measure.label(self.eyelid)
        eye1_ys, eye1_xs = np.where(label_map == 1)
        eye2_ys, eye2_xs = np.where(label_map == 2)

        eye1 = (np.min(eye1_ys) - 10, np.min(eye1_xs) - 10,
                np.max(eye1_ys) + 10, np.max(eye1_xs) + 10)
        eye2 = (np.min(eye2_ys) - 10, np.min(eye2_xs) - 10,
                np.max(eye2_ys) + 10, np.max(eye2_xs) + 10)

        if eye1[1] < eye2[1]: # 1在照片左， 2在照片右
            right_eye = eye1
            left_eye = eye2
        else:
            right_eye = eye2
            left_eye = eye1

        return left_eye, right_eye

    def overlap(self, file):

        right_color = trans_map2color(self.right_eye)
        left_color = trans_map2color(self.left_eye)

        face_map = np.zeros(self.face_img.shape, dtype=np.uint8)
        face_map[self.right_box[0]:self.right_box[2]+1, self.right_box[1]:self.right_box[3]+1] = right_color
        face_map[self.left_box[0]:self.left_box[2]+1, self.left_box[1]:self.left_box[3]+1] = left_color
        face_map = cv2.circle(face_map, (int(self.left_pupil[1]), int(self.left_pupil[0])), 2, (255, 0, 0), 4)
        face_map = cv2.circle(face_map, (int(self.right_pupil[1]), int(self.right_pupil[0])), 2, (255, 0, 0), 4)

        face_map = cv2.addWeighted(self.face_img, 0.6, face_map, 0.4, gamma=0)
        cv2.imwrite(file, face_map)

        return face_map

    def extract_scale(self):
        left_iris = np.zeros_like(self.left_eye, dtype=np.uint8)
        right_iris = np.zeros_like(self.right_eye, dtype=np.uint8)
        left_iris_bound = np.zeros_like(self.left_eye, dtype=np.uint8)
        right_iris_bound = np.zeros_like(self.right_eye, dtype=np.uint8)
        left_iris[self.left_eye == 2] = 255
        right_iris[self.right_eye == 2] = 255
        left_iris_ctrs, _ = cv2.findContours(left_iris, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        right_iris_ctrs, _ = cv2.findContours(right_iris, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left_iris_bound, left_iris_ctrs, -1, 255, 1)
        cv2.drawContours(right_iris_bound, right_iris_ctrs, -1, 255, 1)
        left_iris_ys, left_iris_xs = np.where(left_iris_bound == 255)
        left_iris_dist = compute_dist_and_grad(left_iris_xs, left_iris_ys)
        left_iris_pixel_d = np.max(left_iris_dist)
        left_scale = 11.7 / left_iris_pixel_d

        right_iris_ys, right_iris_xs = np.where(right_iris_bound == 255)
        print(right_iris_ys)
        right_iris_dist = compute_dist_and_grad(right_iris_xs, right_iris_ys)
        right_iris_pixel_d = np.max(right_iris_dist)
        right_scale = 11.7 / right_iris_pixel_d

        return left_iris_pixel_d, right_iris_pixel_d, (left_scale + right_scale) / 2

    def rotate_face(self, eyelid_file, iris_file):
        eyelid_img = cv2.imread(eyelid_file)
        iris_img = cv2.imread(iris_file)
        new_face, _ = rotate(self.face_img, self.right_pupil, self.left_pupil)
        new_eyelid, _ = rotate(eyelid_img, self.right_pupil, self.left_pupil)
        new_iris, _ = rotate(iris_img, self.right_pupil, self.left_pupil)

        return new_face, new_eyelid, new_iris

    def compute_mrd(self):
        left_pupil = (self.left_pupil[0] - self.left_box[0],
                      self.left_pupil[1] - self.left_box[1])
        right_pupil = (self.right_pupil[0] - self.right_box[0],
                       self.right_pupil[1] - self.right_box[1])
        print(self.right_pupil, self.left_pupil)
        print(self.right_box, self.left_box)
        print(right_pupil, left_pupil)
        left_mrd1, left_mrd2, left_pf = calc_mrd(self.left_eye, left_pupil)
        right_mrd1, right_mrd2, right_pf = calc_mrd(self.right_eye, right_pupil)

        return left_mrd1, left_mrd2, left_pf, right_mrd1, right_mrd2, right_pf

    def getPartition(self):
        def get_cornor(eyelid_img, box, side="left"):
            assert side in ["left", "right"]
            ymin, xmin, ymax, xmax = tuple(box)
            single_eyelid = eyelid_img[ymin:ymax+1, xmin:xmax+1]
            eyelid_ys, eyelid_xs = np.where(single_eyelid == 1)
            eyelid_xmin, eyelid_xmax = np.min(eyelid_xs), np.max(eyelid_xs)
            min_slice = single_eyelid[:, eyelid_xmin]
            max_slice = single_eyelid[:, eyelid_xmax]
            min_ys = np.where(min_slice > 0)[0]
            max_ys = np.where(max_slice > 0)[0]
            min_corner = (np.min(min_ys) + ymin, eyelid_xmin + xmin)
            max_corner = (np.min(max_ys) + ymin, eyelid_xmax + xmin)
            if side == "left":   # 左眼，上睑缘为1，下为3，在图像右侧，内眦为xmin， 外眦为xmax
                neizi_coord = min_corner
                waizi_coord = max_corner
            else:
                neizi_coord = max_corner
                waizi_coord = min_corner

            return neizi_coord, waizi_coord

        def get_eyelid_box(eye_box, neizi_coord, waizi_coord):
            # 获取下睑缘的bounding box坐标，其中x坐标为内眦和外眦x，
            # ymin为内眦和外眦中最小的，ymax为eyelid边界框最大y，即y-10
            neizi_y, neizi_x = neizi_coord
            waizi_y, waizi_x = waizi_coord
            xmin = min(neizi_x, waizi_x)
            xmax = max(neizi_x, waizi_x)
            ymin = min(neizi_y, waizi_y)
            ymax = eye_box[2] - 10

            return ymin, xmin, ymax, xmax

        left_neizi_coord, left_waizi_coord = get_cornor(eyelid_img=self.eyelid, box=self.left_box, side="left")
        right_neizi_coord, right_waizi_coord = get_cornor(eyelid_img=self.eyelid, box=self.right_box, side="right")
        left_bottom_eyelid_box = get_eyelid_box(eye_box=self.left_box,
                                                neizi_coord=left_neizi_coord, waizi_coord=left_waizi_coord)
        right_bottom_eyelid_box = get_eyelid_box(eye_box=self.right_box,
                                                 neizi_coord=right_neizi_coord, waizi_coord=right_waizi_coord)
        left_bottom_eyelid_img = None
        left_neizi_img = None
        right_bottom_eyelid_img = None
        right_neizi_img = None

        if left_bottom_eyelid_box:
            left_bottom_eyelid_img = self.face_img[left_bottom_eyelid_box[0]: left_bottom_eyelid_box[2]+129,
                                                   left_bottom_eyelid_box[1]-32: left_bottom_eyelid_box[3]+33, :]
            left_neizi_img = self.face_img[left_neizi_coord[0]-64:left_neizi_coord[0]+65,
                                           left_neizi_coord[1]-64:left_neizi_coord[1]+65, :]

        if right_bottom_eyelid_box:
            right_bottom_eyelid_img = self.face_img[right_bottom_eyelid_box[0]: right_bottom_eyelid_box[2]+129,
                                                    right_bottom_eyelid_box[1]-32: right_bottom_eyelid_box[3]+33, :]
            right_neizi_img = self.face_img[right_neizi_coord[0]-64:right_neizi_coord[0]+65,
                                            right_neizi_coord[1]-64:right_neizi_coord[1]+65, :]

        return left_bottom_eyelid_img, left_neizi_img, right_bottom_eyelid_img, right_neizi_img


if __name__ == "__main__":
    import cv2
    import warnings
    import traceback
    import pandas as pd
    from build_eyelid_param import ManualLabel
    from build_ptosis_param import PtosisFace

    warnings.filterwarnings("ignore")
    # clss = ["blepharostenosis", "ca", "extropion", "TAO", "trichiasis"]
    clss = ["re_trichiasis"]
    df = pd.read_excel("F:/eyelid_data/label/param/re_trichiasis2.xlsx", header=0)
    exist_files = df["fname"].tolist()
    # print(exist_files)
    # print(exist_files)
    for CLS in clss:
        PATH = "F:/eyelid_data/"
        ORIGIN_PATH = PATH + "origin/{}".format(CLS)
        LABEL_PATH = PATH + "label/manual_json/{}".format(CLS)
        FACE_SEG_PATH = PATH + "face_seg/{}".format(CLS)
        VISION_PATH = PATH + "label/partition/{}".format(CLS)
        CLS_DF_FILE = PATH + "label/{}.xlsx".format(CLS)
        if not os.path.exists(VISION_PATH):
            os.makedirs(VISION_PATH)

        flist = os.listdir(ORIGIN_PATH)
        with open("../data/re_trichiasis_failtures.csv", "r") as reader:
            blacklist = [i.rstrip() for i in reader.readlines()]
        for f in flist:
            fname = os.path.splitext(f)[0]
            """
            if not fname in blacklist:
                continue
            """
            face_seg_file = os.path.join(FACE_SEG_PATH, fname + ".npy")
            img_file = os.path.join(ORIGIN_PATH, f)
            manual_label_file = os.path.join(LABEL_PATH, fname + ".json")
            if fname in exist_files:
                continue
            elif fname not in blacklist:
                continue
            try:
                if os.path.exists(face_seg_file):
                    print(fname)
                    eyelid_file = face_seg_file.replace(".npy", "_eyelid.png")
                    iris_file = face_seg_file.replace(".npy", "_iris.png")
                    face = ReTrichData(img_file, eyelid_file, iris_file, manual_json=manual_label_file)
                    left_iris_d, right_iris_d, scale = face.extract_scale()

                    left_mrd1, left_mrd2, left_pf, right_mrd1, right_mrd2, right_pf = face.compute_mrd()
                    tmp_df = pd.DataFrame(data={"fname": fname,
                                                "left_iris": left_iris_d * scale,
                                                "left_mrd1": left_mrd1 * scale,
                                                "left_mrd2": left_mrd2 * scale,
                                                "left_pf": left_pf * scale,
                                                "right_iris": right_iris_d * scale,
                                                "right_mrd1": right_mrd1 * scale,
                                                "right_mrd2": right_mrd2 * scale,
                                                "right_pf": right_pf * scale,
                                                "scale": scale}, index=[0])
                    df = pd.concat([df, tmp_df], ignore_index=True)
                    df.to_excel("F:/eyelid_data/label/param/re_trichiasis.xlsx", index=False)
                    """
                    left_bottom_eyelid_img, left_neizi_img, \
                    right_bottom_eyelid_img, right_neizi_img = face.getPartition()
                    part_path = os.path.join(VISION_PATH, fname)
                    if not os.path.exists(part_path):
                        os.makedirs(part_path)
                    if left_bottom_eyelid_img is not None:
                        cv2.imwrite(os.path.join(part_path, "lb_eyelid.png"), left_bottom_eyelid_img)
                    if left_neizi_img is not None:
                        cv2.imwrite(os.path.join(part_path, "l_neizi.png"), left_neizi_img)
                    if right_bottom_eyelid_img is not None:
                        cv2.imwrite(os.path.join(part_path, "rb_eyelid.png"), right_bottom_eyelid_img)
                    if right_neizi_img is not None:
                        cv2.imwrite(os.path.join(part_path, "r_neizi.png"), right_neizi_img)
                    """
                else:
                    print(fname)
                    face = ManualLabel(img_file, manual_label_file)
                    left_iris_d, right_iris_d, scale = face.calcMrd()

                    left_mrd1, left_mrd2, left_pf, right_mrd1, right_mrd2, right_pf = face.calcMrd()
                    tmp_df = pd.DataFrame(data={"fname": fname,
                                                "left_iris": left_iris_d * scale,
                                                "left_mrd1": left_mrd1 * scale,
                                                "left_mrd2": left_mrd2 * scale,
                                                "left_pf": left_pf * scale,
                                                "right_iris": right_iris_d * scale,
                                                "right_mrd1": right_mrd1 * scale,
                                                "right_mrd2": right_mrd2 * scale,
                                                "right_pf": right_pf * scale,
                                                "scale": scale}, index=[0])
                    df = pd.concat([df, tmp_df], ignore_index=True)
                    df.to_excel("F:/eyelid_data/label/param/re_trichiasis.xlsx", index=False)
            except Exception:
                print(fname)
                print(traceback.format_exc())
                continue

    # df.to_excel("F:/eyelid_data/label/param/re_trichiasis.xlsx", index=False)