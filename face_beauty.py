from models.detector import face_detector
import numpy as np
from models.parser import face_parser
import cv2, os

part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
               [255, 0, 85], [255, 0, 170],
               [0, 255, 0], [85, 255, 0], [170, 255, 0],
               [0, 255, 85], [0, 255, 170],
               [0, 0, 255], [85, 0, 255], [170, 0, 255],
               [0, 85, 255], [0, 170, 255],
               [255, 255, 0], [255, 255, 85], [255, 255, 170],
               [255, 0, 255], [255, 85, 255], [255, 170, 255],
               [0, 255, 255], [85, 255, 255], [170, 255, 255]]


def show_face_bbox(img_path):
    """
    detecting face bbox.
    :param img_path: img
    :return:
    """
    if not os.path.exists("./result"):
        print("make dir!")
        os.mkdir("./result")

    im = cv2.imread(img_path)
    fd = face_detector.FaceAlignmentDetector()
    bboxes = fd.detect_face(im, with_landmarks=False)
    ret = bboxes[0][0:4]
    print(ret)

    cv2.rectangle(im, (int(ret[1]), int(ret[0])), (int(ret[3]), int(ret[2])), (0, 255, 0), 2)
    score = bboxes[0][-1]

    cv2.imwrite("./result/test_bbox.jpg", im)


def show_face_parser(img_path, save_img=True):
    """
    facial segmentation.
    :param img_path:
    :return:
    """
    im = cv2.imread(img_path)
    print(im.shape)
    h, w = im.shape[0:2]
    fp = face_parser.FaceParser()
    # fp.set_detector(fd) # fd = face_detector.FaceAlignmentDetector()
    parsing_map = fp.parse_face(im, bounding_box=None, with_detection=False)
    map = parsing_map[0].reshape(h, w, 1)

    mask1 = map == 10
    mask2 = map == 1
    mask3 = map == 14
    mask = (mask1 + mask2 + mask3).astype(np.uint8)
    # mask = cv2.GaussianBlur(mask, (5, 5), 0)
    img_mask_fg = cv2.bitwise_and(im, im, mask=mask)

    mask_inv = cv2.bitwise_not(mask * 255)
    # mask_inv = cv2.GaussianBlur(mask_inv, (5, 5), 0)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    img_mask_bg = cv2.bitwise_and(im, im, mask=mask_inv)

    num_of_class = 17
    if save_img:
        map = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)

        map_color = np.zeros_like(map)

        for pi in range(1, num_of_class + 1):
            # print(pi, part_colors[pi])
            index = np.where(map == pi)
            map_color[index[0], index[1], :] = part_colors[pi]

        cv2.imwrite("./result/test_seg.jpg", map_color)
        cv2.imwrite("./result/test_mask.jpg", mask * 255)
        cv2.imwrite("./result/img_mask_fg.jpg", img_mask_fg)
        cv2.imwrite("./result/img_mask_bg.jpg", img_mask_bg)
        print("Mask saved!")
    return img_mask_fg, img_mask_bg, mask


def fast_guideFilter(I, p, winSize, eps, s):
    """
    Fast guidedFilter
    :param I:
    :param p:
    :param winSize:
    :param eps:
    :param s:
    :return:
    """
    h, w = I.shape[:2]

    size = (int(round(w * s)), int(round(h * s)))
    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)

    X = winSize[0]
    small_winSize = (int(round(X * s)), int(round(X * s)))

    mean_small_I = cv2.blur(small_I, small_winSize)
    mean_small_p = cv2.blur(small_p, small_winSize)

    mean_small_II = cv2.blur(small_I * small_I, small_winSize)
    mean_small_Ip = cv2.blur(small_I * small_p, small_winSize)

    var_small_I = mean_small_II - mean_small_I * mean_small_I
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p
    # print(var_small_I.mean())
    # print(cov_small_Ip.mean())
    # if var_small_I.mean() >= 0.009:
    #     eps = 0.01

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a * mean_small_I

    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)

    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

    q = mean_a * I + mean_b

    return q


def guideFilter(img):
    """

    :param img:
    :return:
    """
    guide = img
    # guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dst1 = cv2.ximgproc.guidedFilter(
        guide=guide, src=img, radius=32, eps=2000, dDepth=-1)
    dst2 = cv2.ximgproc.guidedFilter(
        guide=guide, src=img, radius=64, eps=1000, dDepth=-1)
    dst3 = cv2.ximgproc.guidedFilter(
        guide=guide, src=img, radius=32, eps=1000, dDepth=-1)

    return dst1, dst2, dst3

if __name__ == '__main__':
    img_path = "./1.jpeg"
    fg, bg, mask_fg = show_face_parser(img_path, True)
    ## guided filter
    # dst1, dst2, dst3 = guideFilter(fg)
    #
    # dst1 = cv2.add(dst1, bg)
    # dst2 = cv2.add(dst2, bg)
    # dst3 = cv2.add(dst3, bg)
    #
    #
    # cv2.imwrite("./result/image_eps50.jpg", dst1)
    # cv2.imwrite("./result/image_eps500.jpg", dst2)
    # cv2.imwrite("./result/image_eps1000.jpg", dst3)

    ## Fast guilded filter
    gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)

    var = cv2.meanStdDev(gray, mask=mask_fg)
    print(var)
    eps = 0.001 if var[1] < 40 else 0.01
    print(eps)
    winSize = (16, 16)  # convolution kernel

    # image = cv2.resize(fg, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
    I = fg / 255.0  #
    p = I
    s = 3  # step length

    guideFilter_img = fast_guideFilter(I, p, winSize, eps, s)
    guideFilter_img = guideFilter_img * 255  # (0,1)->(0,255)
    guideFilter_img[guideFilter_img > 255] = 255
    guideFilter_img = np.round(guideFilter_img)
    guideFilter_img = guideFilter_img.astype(np.uint8)
    guideFilter_img = cv2.add(guideFilter_img, bg)

    img_zero = np.zeros_like(fg)
    ret, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_zero, contours, -1, (255, 255, 255), 3)

    blurred_img = guideFilter_img

    output = np.where(img_zero == np.array([255, 255, 255]), cv2.GaussianBlur(blurred_img, (5, 5), 0), blurred_img)
    cv2.imwrite("./result/mask.jpg", img_zero)
    cv2.imwrite("./result/post.jpg", output)
    cv2.imwrite("./result/winSize_16.jpg", guideFilter_img)
