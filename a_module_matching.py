import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops, regionprops_table
import datetime

def get_mask(img,hsv_range):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h_min, h_max, s_min, s_max, v_min, v_max = 55, 135, 45, 255, 70, 255
    lower = np.array([hsv_range[0], hsv_range[2], hsv_range[4]])
    upper = np.array([hsv_range[1], hsv_range[3], hsv_range[5]])
    mask = cv2.inRange(imgHSV, lower, upper)
    # 滤波去除噪声, 小于5个像素的区域直接移除
    # viewImage(mask)
    #不腐蚀之前 可分割小组件
    kernel = np.ones((3, 3), np.uint8)
    #腐蚀 是缩小
    mask = cv2.erode(mask, kernel, 2)
    # viewImage(mask)
    #膨胀
    mask = cv2.dilate(mask, kernel, iterations=6)
    # viewImage(mask)
    return mask

def get_region(mask):
    label_im = label(mask)
    regions = regionprops(label_im)
    return regions
# 计算欧式距离
def eucldist_forloop(coords1, coords2):

    """ Calculates the euclidean distance between 2 lists of coordinates. """
    dist = 0
    for (x, y) in zip(coords1, coords2):
        dist += (x - y)**2
    return dist**0.5

def plt_module(img,num,centroid):
    draw_img = img.copy()
    cv2.putText(draw_img, "#{}".format(num), (int(centroid[1]), int(centroid[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                3, (255, 0, 255), 3)
    plt.imshow(draw_img)
    plt.show()
    return draw_img

def get_target_module(target_pixel,sigle_region,mosaic_region):
    # 计算目标像素在单幅图中的组串标号以及质心位置
    for num, x in enumerate(sigle_region):
        if x.area < 100000:
            continue
        bbox = x.bbox
        if target_pixel[0] > bbox[1] and target_pixel[0] < bbox[3] and target_pixel[1] > bbox[0] and target_pixel[1] < bbox[2]:
             single_num = num
             single_centroid = sigle_region[num].centroid
    #根据单幅图中的目标组件质心得出拼接图中的对应质心
    min_dis = 9999
    for num2, x2 in enumerate(mosaic_region):
        dis = eucldist_forloop(single_centroid, x2.centroid)
        if dis < min_dis:
            min_dis = dis
            mosaic_num = num2
    return sigle_region[single_num],mosaic_region[mosaic_num]

if __name__ == "__main__" :
    sigle_img_path = './images/DJI_20210803105452_0067_W.JPG'
    mosaic_img_path = './result/output_crop_raster_0067.tif'
    single_img = cv2.imread(sigle_img_path)
    mosaic_img = cv2.imread(mosaic_img_path)
    #   ----------------阈值分割
    sigle_mask = get_mask(single_img, hsv_range=[55, 135, 45, 255, 70, 255])
    # 差别主要在于亮度值v
    mosaic_mask = get_mask(mosaic_img, hsv_range=[55, 135, 45, 255, 90, 255])

    # -------------通过regionprops函数获取分割区域-------------------
    mosaic_list_of_region = get_region(mosaic_mask)
    sigle_list_of_region = get_region(sigle_mask)

    # -------------计算对应拼接图中的组件串-------------------
    # 目标像素点
    target_pixel = [2019, 1401]
    single_region,mosaic_region = get_target_module(target_pixel, sigle_list_of_region,
                                                                       mosaic_list_of_region)
    # 显示结果
    single_pos_img = plt_module(single_img,single_region.label,single_region.centroid)
    mosaic_pos_img = plt_module(mosaic_img,mosaic_region.label,mosaic_region.centroid)
    cv2.imwrite('single_'+str(single_region.label)+'.jpg',single_pos_img)
    cv2.imwrite('mosaic_' + str(mosaic_region.label) + '.jpg', mosaic_pos_img)