import os
import cv2
import json
import numpy as np
import pandas as pd

from utils import fill, calc_mrd, compute_dist_and_grad, get_long_diameter_idx


class ManualLabel(object):
    def __init__(self, face_file, manual_json):
        self.face_img = cv2.imread(face_file)
        self.center_x = self.face_img.shape[2] / 2
        self.orgs = self.parseJson(manual_json)  # 均为xy顺序
        print(self.orgs["pupils"])
        self.eyelid_map, self.iris_map = self.buildLabelMap()
        self.left_iris, self.right_iris, self.scale = self.calcScale()

    def parseJson(self, manual_json):
        with open(manual_json, "r") as obj:
            js_data = json.load(obj)
        orgs = js_data["shapes"]
        ps = []
        uls = []
        lls = []
        iriss = []
        for org in orgs:
            label = org["label"]
            points = org["points"]
            if label == "p":
                ps.append(points)
            elif label == "ul":
                uls.append(points)
            elif label == "ll":
                lls.append(points)
            elif label == "iris":
                iriss.append(points)
        p_dict = self.getOrgdict(ps)
        ul_dict = self.getOrgdict(uls)
        ll_dict = self.getOrgdict(lls)
        iris_dict = self.getOrgdict(iriss)
        org_dict = {"pupils": p_dict,
                    "upper_eyelids": ul_dict,
                    "lower_eyelids": ll_dict,
                    "iriss": iris_dict}

        return org_dict

    def getOrgdict(self, orgs):
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
                org_dict = {"right_eye": orgs[0], "left_eye": orgs[1]}
            else:
                org_dict = {"right_eye": orgs[1], "left_eye": orgs[0]}
        else:
            if self.judgeLR(coords):
                org_dict = {"right_eye": orgs[0]}
            else:
                org_dict = {"left_eye": orgs[0]}

        return org_dict

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

    def buildLabelMap(self):
        eyelid_map = np.zeros((self.face_img.shape[0], self.face_img.shape[1]), dtype=np.int32)
        iris_map = np.zeros((self.face_img.shape[0], self.face_img.shape[1]), dtype=np.int32)

        def draw_line(coords, c):
            for i in range(len(coords) - 1):
                start = (int(coords[i][0]), int(coords[i][1]))
                end = (int(coords[i + 1][0]), int(coords[i + 1][1]))
                cv2.line(img=eyelid_map, pt1=start, pt2=end, color=c, thickness=1)

        def draw_circle(coords, c):
            for i in range(len(coords)):
                start = (int(coords[i][0]), int(coords[i][1]))
                if i == len(coords) - 1:
                    end = (int(coords[0][0]), int(coords[0][1]))
                else:
                    end = (int(coords[i+1][0]), int(coords[i+1][1]))
                cv2.line(img=iris_map, pt1=start, pt2=end, color=c, thickness=1)

        for org, locs in self.orgs.items():
            if org == "upper_eyelids":
                for part, coords in locs.items():
                    if part == "left_eye":
                        draw_line(coords, 1)  # 左眼上睑缘为1
                    else:
                        draw_line(coords, 2)  # 右眼上睑缘为2
            elif org == "lower_eyelids":
                for part, coords in locs.items():
                    if part == "left_eye":
                        draw_line(coords, 3)  # 左眼下睑缘为3
                    else:
                        draw_line(coords, 4)  # 右眼下睑缘为4
            elif org == "iriss":
                for part, coords in locs.items():
                    if part == "left_eye":
                        draw_circle(coords, 5)   # 左眼虹膜为5
                    else:
                        draw_circle(coords, 6)   # 右眼虹膜为6

        return eyelid_map, iris_map

    def outputLabelMap(self, path):
        color_label_map = np.zeros_like(self.face_img, dtype=np.uint8)
        color_label_map[np.where(self.eyelid_map == 1)] = (255, 0, 0)  # 左眼上睑缘蓝色
        color_label_map[np.where(self.eyelid_map == 2)] = (0, 255, 0)  # 右眼上睑缘绿色
        color_label_map[np.where(self.eyelid_map == 3)] = (0, 0, 255)  # 左眼下睑缘红色
        color_label_map[np.where(self.eyelid_map == 4)] = (255, 255, 0)  # 右眼下睑浅蓝
        color_label_map[np.where(self.eyelid_map == 5)] = (0, 255, 255)  # 右眼下睑浅蓝
        color_label_map[np.where(self.eyelid_map == 6)] = (255, 255, 255)  # 右眼下睑浅蓝

        cv2.imwrite(path, color_label_map)

    def calcScale(self):
        real_len = 11.7
        left_scale, right_scale = None, None
        left_iris_ys, left_iris_xs = np.where(self.iris_map == 5)
        left_iris_pixel_d, right_iris_pixel_d = 0, 0
        if left_iris_xs.shape[0] > 1:
            left_iris_dist = compute_dist_and_grad(left_iris_xs, left_iris_ys)
            left_iris_pixel_d = np.max(left_iris_dist)
            left_scale = real_len / left_iris_pixel_d
        right_iris_ys, right_iris_xs = np.where(self.iris_map == 6)
        if right_iris_xs.shape[0] > 1:
            right_iris_dist = compute_dist_and_grad(right_iris_xs, right_iris_ys)
            right_iris_pixel_d = np.max(right_iris_dist)
            right_scale = real_len / right_iris_pixel_d

        if left_scale is not None and right_scale is not None:
            scale = (left_scale + right_scale) / 2
            return left_iris_pixel_d * scale, right_iris_pixel_d * scale, scale
        elif left_scale is not None and right_scale is None:
            scale = left_scale
            return left_iris_pixel_d * scale, right_iris_pixel_d * scale, left_scale
        elif right_scale is not None and left_scale is None:
            scale = right_scale
            return left_iris_pixel_d * scale, right_iris_pixel_d * scale, right_scale
        else:
            return left_iris_pixel_d, right_iris_pixel_d, None

    def calcMrd(self):
        pupils = self.orgs["pupils"]
        eye_params = {}
        if "left_eye" in pupils:
            left_pupil = pupils["left_eye"][0]  # x y
            left_pixel_mrd1, left_pixel_mrd2, left_pixel_pf = calc_mrd(pupil_coord=left_pupil,
                                                                       eyelid_map=self.eyelid_map,
                                                                       side="left")
            eye_params["left_eye"] = (left_pixel_mrd1 * self.scale,
                                      left_pixel_mrd2 * self.scale,
                                      left_pixel_pf * self.scale)
        else:
            eye_params["left_eye"] = (0, 0, 0)
        if "right_eye" in pupils:
            right_pupil = pupils["right_eye"][0]
            right_pixel_mrd1, right_pixel_mrd2, right_pixel_pf = calc_mrd(pupil_coord=right_pupil,
                                                                          eyelid_map=self.eyelid_map,
                                                                          side="right")
            eye_params["right_eye"] = (right_pixel_mrd1 * self.scale,
                                       right_pixel_mrd2 * self.scale,
                                       right_pixel_pf * self.scale)
        else:
            eye_params["right_eye"] = (0, 0, 0)

        return eye_params

    def getPartition(self):
        def get_bbox(img, id):
            ys, xs = np.where(img == id)
            if ys.shape[0] > 0:
                ymin, ymax = np.min(ys), np.max(ys)
                xmin, xmax = np.min(xs), np.max(xs)

                return ymin, xmin, ymax, xmax
            else:
                return None

        def get_cornor(img, xmin, xmax, side="left"):
            assert side in ["left", "right"]
            min_slice = img[:, xmin]
            min_ys = np.where(min_slice > 0)[0]
            max_slice = img[:, xmax]
            max_ys = np.where(max_slice > 0)[0]
            min_corner = (np.min(min_ys), xmin)
            max_corner = (np.min(max_ys), xmax)
            if side == "left":   # 左眼，上睑缘为1，下为3，在图像右侧，内眦为xmin， 外眦为xmax
                neizi_coord = min_corner
                waizi_coord = max_corner
            else:
                neizi_coord = max_corner
                waizi_coord = min_corner

            return neizi_coord, waizi_coord

        left_top_eyelid_box = get_bbox(self.eyelid_map, id=1)
        left_bottom_eyelid_box = get_bbox(self.eyelid_map, id=3)
        right_top_eyelid_box = get_bbox(self.eyelid_map, id=2)
        right_bottom_eyelid_box = get_bbox(self.eyelid_map, id=4)
        left_top_eyelid_img, left_bottom_eyelid_img = None, None
        left_neizi_img, left_waizi_img = None, None
        right_top_eyelid_img, right_bottom_eyelid_img = None, None
        right_neizi_img, right_waizi_img = None, None

        if left_top_eyelid_box:
            left_neizi_coord, left_waizi_coord = get_cornor(self.eyelid_map,
                                                            left_top_eyelid_box[1],
                                                            left_top_eyelid_box[3],
                                                            side="left")
            left_top_eyelid_img = self.face_img[left_top_eyelid_box[0]-128: left_top_eyelid_box[2]+1,
                                                left_top_eyelid_box[1]-32: left_top_eyelid_box[3]+33, :]
            left_neizi_img = self.face_img[left_neizi_coord[0]-64:left_neizi_coord[0]+65,
                                           left_neizi_coord[1]-64:left_neizi_coord[1]+65, :]
            left_waizi_img = self.face_img[left_waizi_coord[0]-64:left_waizi_coord[0]+65,
                                           left_waizi_coord[1]-64:left_waizi_coord[1]+65, :]
        if left_bottom_eyelid_box:
            left_bottom_eyelid_img = self.face_img[left_bottom_eyelid_box[0]: left_bottom_eyelid_box[2]+129,
                                                   left_bottom_eyelid_box[1]-32: left_bottom_eyelid_box[3]+33, :]

        if right_top_eyelid_box:
            right_neizi_coord, right_waizi_coord = get_cornor(self.eyelid_map,
                                                               right_top_eyelid_box[1],
                                                               right_top_eyelid_box[3],
                                                               side="right")
            right_top_eyelid_img = self.face_img[right_top_eyelid_box[0]-128: right_top_eyelid_box[2]+1,
                                                 right_top_eyelid_box[1]-32: right_top_eyelid_box[3]+33, :]
            right_neizi_img = self.face_img[right_neizi_coord[0]-64:right_neizi_coord[0]+65,
                                            right_neizi_coord[1]-64:right_neizi_coord[1]+65, :]
            right_waizi_img = self.face_img[right_waizi_coord[0]-64:right_waizi_coord[0]+65,
                                            right_waizi_coord[1]-64:right_waizi_coord[1]+65, :]
        if right_bottom_eyelid_box:
            right_bottom_eyelid_img = self.face_img[right_bottom_eyelid_box[0]: right_bottom_eyelid_box[2]+129,
                                                    right_bottom_eyelid_box[1]-32: right_bottom_eyelid_box[3]+33, :]

        return left_top_eyelid_img, left_bottom_eyelid_img, left_neizi_img, left_waizi_img, \
               right_top_eyelid_img, right_bottom_eyelid_img, right_neizi_img, right_waizi_img


if __name__ == "__main__":
    import warnings
    import traceback

    warnings.filterwarnings("ignore")
    # clss = ["blepharostenosis", "ca", "extropion", "TAO", "trichiasis"]
    clss = ["extropion"]
    for CLS in clss:
        print(CLS)
        PATH = "E:/eyelid_data/"
        ORIGIN_PATH = PATH + "origin/{}".format(CLS)
        LABEL_PATH = PATH + "label/manual_json/{}".format(CLS)
        VISION_PATH = PATH + "label/partition/{}".format(CLS)
        CLS_DF_FILE = PATH + "label/{}.xlsx".format(CLS)
        if not os.path.exists(VISION_PATH):
            os.makedirs(VISION_PATH)

        flist = os.listdir(ORIGIN_PATH)
        for f in flist:
            img_file = os.path.join(ORIGIN_PATH, f)
            fname = os.path.splitext(f)[0]
            msk_file = os.path.join(LABEL_PATH, fname + ".json")
            if not "IMG" in fname:
                continue
            try:
                if os.path.exists(msk_file):
                    face = ManualLabel(img_file, msk_file)
                    scale = face.scale
                    eye_param = face.calcMrd()
                    left_top_eyelid_img, left_bottom_eyelid_img, \
                    left_neizi_img, left_waizi_img, \
                    right_top_eyelid_img, right_bottom_eyelid_img, \
                    right_neizi_img, right_waizi_img = face.getPartition()
                    part_path = os.path.join(VISION_PATH, fname)
                    if not os.path.exists(part_path):
                        os.makedirs(part_path)
                    if left_top_eyelid_img is not None:
                        cv2.imwrite(os.path.join(part_path, "lt_eyelid.png"), left_top_eyelid_img)
                    if left_bottom_eyelid_img is not None:
                        cv2.imwrite(os.path.join(part_path, "lb_eyelid.png"), left_bottom_eyelid_img)
                    if left_neizi_img is not None:
                        cv2.imwrite(os.path.join(part_path, "l_neizi.png"), left_neizi_img)
                    if left_waizi_img is not None:
                        cv2.imwrite(os.path.join(part_path, "l_waizi.png"), left_waizi_img)
                    if right_top_eyelid_img is not None:
                        cv2.imwrite(os.path.join(part_path, "rt_eyelid.png"), right_top_eyelid_img)
                    if right_bottom_eyelid_img is not None:
                        cv2.imwrite(os.path.join(part_path, "rb_eyelid.png"), right_bottom_eyelid_img)
                    if right_neizi_img is not None:
                        cv2.imwrite(os.path.join(part_path, "r_neizi.png"), right_neizi_img)
                    if right_waizi_img is not None:
                        cv2.imwrite(os.path.join(part_path, "r_waizi.png"), right_waizi_img)
            except cv2.error:
                print(f)
