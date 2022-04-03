from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import cv2
# from draw_matches import draw_matches
from enum import Enum
# parser = argparse.ArgumentParser(description='Code from AKAZE local features matching tutorial.')
# parser.add_argument('--input1', help='Path to input image 1.', default='graf1.png')
# parser.add_argument('--input2', help='Path to input image 2.', default='graf3.png')
# parser.add_argument('--homography', help='Path to the homography matrix.', default='H1to3p.xml')
# args = parser.parse_args()

class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5

# Load grayscale images
img1 = mpimg.imread(r"E:\xlq\PycharmProjects\LearnCV\TIFF_MATCH\ImageRegistration\images\DJI_0452-tryout-defishr.jpg")
img2 = mpimg.imread(r'E:\xlq\PycharmProjects\LearnCV\TIFF_MATCH\ImageRegistration\images\MOSAIC_YC.tif')

#复制一些原图的副本，画图用
img1_list = []
img2_list = []
for i in range(15):
    img1_list.append(img1.copy())
    img2_list.append(img2.copy())

img11 = img1.copy()
img22 = img2.copy()


img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

figure, ax = plt.subplots(1, 2, figsize=(16, 8))

ax[0].imshow(img1_g, cmap='gray')
ax[1].imshow(img2_g, cmap='gray')
plt.show()


# fs = cv.FileStorage(args.homography, cv.FILE_STORAGE_READ)
# homography = fs.getFirstTopLevelNode().mat()

#ORB detector
orb = cv2.ORB_create(10000)
kps1 = orb.detect(img1_g,None)
kps2 = orb.detect(img2_g,None)

#BEBLID descriptor
beblid = cv2.xfeatures2d.BEBLID_create(0.75)
keypoints_1, descriptors_1 = beblid.compute(img1_g,kps1)
keypoints_2, descriptors_2 = beblid.compute(img2_g,kps2)


#可视化特征点检测结果

red = (255,0,0)
#color = (51,163,236)
cv2.drawKeypoints(image = img1_list[11],
                  outImage = img1_list[11],
                  keypoints = keypoints_1,
                  color = red)
cv2.drawKeypoints(image = img2_list[11],
                  outImage = img2_list[11],
                  keypoints = keypoints_2,
                  color = red)
figure, ax = plt.subplots(1, 2, figsize=(20, 20),dpi=160)
ax[0].imshow(img1_list[11])
ax[1].imshow(img2_list[11])
#第二个参数表示去掉多余空白位置
plt.savefig("./result/ORB+BEBLID_yc_Extract.png", bbox_inches='tight')
plt.show()

descriptors_1 = np.asarray(descriptors_1, np.float32)
descriptors_2 = np.asarray(descriptors_2, np.float32)
# 排除错误匹配点
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches1 = flann.knnMatch(descriptors_1, descriptors_2, k=2)
good_matches1 = []
for m, n in matches1:
    if m.distance < 0.7 * n.distance:
        # if math.fabs(kps1[m.queryIdx].pt[1] - kps2[m.trainIdx].pt[1]) < 20 and math.fabs(kps1[m.queryIdx].pt[0] - kps2[m.trainIdx].pt[0]) < 60:
        # 因为是双目摄像头，视差有一定的限制
        good_matches1.append(m)

good_matches2 = cv2.xfeatures2d.matchGMS((img1.shape[:2][::-1]), (img2.shape[:2][::-1]), keypoints_1, keypoints_2,
                                         good_matches1)
good_kps_l = [keypoints_1[m.queryIdx] for m in good_matches2]
good_kps_r = [keypoints_2[m.trainIdx] for m in good_matches2]

im = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches2, None)
im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
cv2.imwrite('./result/show_match.jpg', im)




img3 = cv2.drawMatchesKnn(img1_list[11],keypoints_1,img2_list[11],keypoints_2,matched,None,flags=2)


#获取两个图对应的特征点
ptsA= np.float32([keypoints_1[m[0].queryIdx].pt for m in matched]).reshape(-1,1,2)
ptsB = np.float32([keypoints_2[m[0].trainIdx].pt for m in matched]).reshape(-1, 1, 2)

ransacReprojThreshold = 4
#  单应性矩阵可以将一张图通过旋转、变换等方式与另一张图对齐
# H就是变换矩阵，status是掩码
H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);

#按单应矩阵变换第二张图,大小为两张图片水平大小
imgOut = cv2.warpPerspective(img1, H, (img2.shape[1],img2.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

imgOut_2 = imgOut.copy()
plt.figure(figsize=(20,20));
plt.imshow(imgOut)
# plt.savefig("./result/ORB_AND_BEBLID_H.png", bbox_inches='tight')
plt.show()

res = cv2.addWeighted(imgOut,0.5,img2,0.5,0)
plt.figure(figsize=(20,20));
plt.imshow(res)
# plt.savefig("./result/ORB_AND_BEBLID_fusion.png", bbox_inches='tight')
plt.show()
#
# imgOut_2[0:img1.shape[0], 0:img1.shape[1]] = img1
#
# plt.figure(figsize=(20,20));
# plt.imshow(imgOut_2)
# plt.savefig("FAST_AND_BRIEF_Stitching.png", bbox_inches='tight')
# plt.show()


