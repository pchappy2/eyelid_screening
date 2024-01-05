
import cv2
import numpy as np


def rotate(img, A, B):
    x1, y1 = A
    x2, y2 = B
    if len(img.shape) == 3:
        rows, cols, _ = img.shape
    else:
        rows, cols = img.shape
    theta = np.arctan((x1 - x2) / (y1 - y2)) / 2 / np.pi * 360

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    return dst, theta


def fill(img_slice):
    """
    水漫填充,类型为np.uint8，需要先转换为[0, 255]二值化
    :param im_in:
    :return:
    """
    im_in = img_slice * 255
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)   # 填充 255
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)     # ---> Modified 1
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv   # [0, 255]二值化

    return im_out / 255


def calc_mrd(pupil_coord, eyelid_map, side="left"):
    assert side in ["left", "right"]
    up_item = 1 if side == "left" else 2
    bottom_item = 3 if side == "left" else 4
    pupil_x = int(pupil_coord[0])
    pupil_y = int(pupil_coord[1])
    seg_slice = eyelid_map[:, pupil_x]
    up_coords = np.where(seg_slice == up_item)[0]
    bottom_coords = np.where(seg_slice == bottom_item)[0]
    up_y = np.min(up_coords)
    bottom_y = np.max(bottom_coords)
    mrd1 = pupil_y - up_y + 1
    mrd2 = bottom_y - pupil_y + 1
    pf = mrd1 + mrd2

    return mrd1, mrd2, pf


def compute_dist_and_grad(xs, ys):
    """
    计算边界上任意两点之间的距离和斜率
    :param xs, ys: ndarray
    :return dist:
    :return grad:
    """

    x_len = xs.shape[0]
    y_len = ys.shape[0]
    xs_0 = np.repeat(xs, x_len)
    ys_0 = np.repeat(ys, y_len)

    xs_1 = np.reshape(xs_0, (x_len, x_len))
    ys_1 = np.reshape(ys_0, (y_len, y_len))

    xs_2 = np.transpose(xs_1)
    ys_2 = np.transpose(ys_1)

    x_offset = np.power(xs_1 - xs_2, 2)
    y_offset = np.power(ys_1 - ys_2, 2)

    dist = np.sqrt(y_offset + x_offset)
    # grad = (ys_2 - ys_1) / (xs_2 - xs_1 + 1e-10)

    return dist


def get_long_diameter_idx(dist):
    """
    计算距离最长两点的index
    :param dist: 两点距离矩阵
    :return:
    """
    idx = np.unravel_index(np.argmax(dist), dist.shape)

    return idx