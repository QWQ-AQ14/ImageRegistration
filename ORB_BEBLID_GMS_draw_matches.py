import numpy as np
from enum import Enum
import time
import cv2
# from cv2.xfeatures2d import matchGMS
import matplotlib.pyplot as plt
# ORB+BEBLID+GMS

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (255, 0, 0))

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 1, (0, 255, 255), 2)
            cv2.circle(output, tuple(map(int, right)), 1, (0, 255, 0), 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1]*.5 + left[1] - src1.shape[0]*.5) * 256. / (src1.shape[0]*.5 + src1.shape[1]*.5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 1, color, 2)
            cv2.circle(output, tuple(map(int, right)), 1, color, 2)
    return output


if __name__ == '__main__':
    img1 = cv2.imread("./images/MOSAIC_YC.tif")
    img2 = cv2.imread("./images/YC4000_big.jp2")
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 特征点检测
    orb = cv2.ORB_create(10000)
    # 快速阈值
    # orb.setFastThreshold(0)
    kps1 = orb.detect(img1_g, None)
    kps2 = orb.detect(img2_g, None)

    # BEBLID descriptor
    beblid = cv2.xfeatures2d.BEBLID_create(0.75)
    kp1, des1 = beblid.compute(img1_g, kps1)
    kp2, des2 = beblid.compute(img2_g, kps2)
    # 建立匹配关系
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # 匹配描述子
    matches_all = matcher.match(des1, des2)


    start = time.time()
    matches_gms = cv2.xfeatures2d.matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=False, withRotation=False, thresholdFactor=6)
    end = time.time()

    print('Found', len(matches_gms), 'matches')
    print('GMS takes', end-start, 'seconds')

    output = draw_matches(img1, img2, kp1, kp2, matches_gms, DrawingType.ONLY_LINES)
    plt.figure(figsize=(20, 20), dpi=160);
    plt.imshow(output)
    plt.savefig("./result/ORB_AND_BEBLID_YC_Match.png", bbox_inches='tight')
    plt.show()

    # 获取两个图对应的特征点
    ptsA = np.float32([kp1[m.queryIdx].pt for m in matches_gms]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m.trainIdx].pt for m in matches_gms]).reshape(-1, 1, 2)

    # ransacReprojThreshold = 4
    #  单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
    # H就是变换矩阵，status是掩码
    H, status = cv2.findHomography(ptsA, ptsB);
    # 按单应矩阵变换第二张图,大小为两张图片水平大小
    # 按单应矩阵变换第二张图,大小为两张图片水平大小
    imgOut = cv2.warpPerspective(img1_g, H, (img2_g.shape[1], img2_g.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    plt.figure(figsize=(20, 20))
    plt.imshow(imgOut)
    # plt.savefig("./result/ORB_AND_BEBLID_H.png", bbox_inches='tight')
    plt.show()

    res = cv2.addWeighted(imgOut, 0.5, img2_g, 0.5, 0)
    plt.figure(figsize=(20, 20));
    plt.imshow(res)
    # plt.savefig("./result/ORB_AND_BEBLID_fusion.png", bbox_inches='tight')
    plt.show()

