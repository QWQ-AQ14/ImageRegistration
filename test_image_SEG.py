# 分割组件进行编号
import cv2
import numpy as np
from scipy.signal import argrelextrema
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table

def viewImage(img):
    cv2.namedWindow('Display', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('Display', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_boundaries(value, tolerance, ranges, upper_or_lower):
    if ranges == 0:
        # set the boundary for hue
        boundary = 180
    elif ranges == 1:
        # set the boundary for saturation and value
        boundary = 255

    if(value + tolerance > boundary):
        value = boundary
    elif (value - tolerance < 0):
        value = 0
    else:
        if upper_or_lower == 1:
            value = value + tolerance
        else:
            value = value - tolerance
    return value

def get_HSV_Hist_polyfit(hist):
    #获取HSV中直方图后进行拟合 得到拟合后的结果
    x = np.arange(0, 256, 1)
    y = hist.flatten()
    z1 = np.polyfit(x, y, 50)
    p1 = np.poly1d(z1)
    # print(p1)  # 在屏幕上打印拟合多项式
    yvals = p1(x)  # 也可以使用yvals=np.polyval(z1,x)
    plot1 = plt.plot(x, y, '*', label='original values')
    plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend(loc=1)  # 指定legend的位置,读者可以自己help它的用法
    plt.title('polyfitting')
    plt.show()
    return yvals

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
    viewImage(mask)
    return mask

# 自动获得HSV的阈值范围
def get_HSV(img):
    # 将图片转换为HSV格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_range = []
    # 按R、G、B三个通道分别计算颜色直方图
    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        val_hist = get_HSV_Hist_polyfit(hist)
        # 获取各个拟合之后的峰值索引
        maxima = argrelextrema(val_hist, np.greater)  # 极大值
        minima = argrelextrema(val_hist, np.less)  # 极小值
        sum_index = list(np.msort(np.append(minima[0], maxima[0])))
        # 获取极大值中最大值的下标
        max_index = np.argmax(val_hist[maxima])
        max_index_num = maxima[0][max_index]
        # 获取极小值中的最小值坐标
        min_index = np.argmin(val_hist[minima])
        min_index_num = minima[0][min_index]
        # max_index = v_val.tolist().index(max(v_val.tolist()))
        minima_arr = np.append(minima[0], [max_index_num])
        # print(np.append(minima[0], [max_index_num]))
        minima_arr = list(np.msort(minima_arr))
        minima_list_max_index = minima_arr.index(max_index_num)
        # print(minima_arr[minima_list_max_index - 1], minima_arr[minima_list_max_index + 1])
        #根据蓝色先验知识进行判断
        if minima_arr[minima_list_max_index - 1] > 30 and minima_arr[minima_list_max_index - 1] < 200:
            hsv_range.append(minima_arr[minima_list_max_index - 1])
            hsv_range.append(minima_arr[minima_list_max_index + 1])
        else:
            i = 0
            while sum_index[minima_list_max_index + i] < 40:
                i +=1
            hsv_range.append(sum_index[minima_list_max_index + i])
            while sum_index[minima_list_max_index + i] < 200:
                i+=1
            hsv_range.append(sum_index[minima_list_max_index + i])

    print(hsv_range)
    #生成二值图
    lower = np.array([hsv_range[0], hsv_range[2], hsv_range[4]])
    upper = np.array([hsv_range[1], hsv_range[3], hsv_range[5]])
    mask = cv2.inRange(img, lower, upper)
    # viewImage(mask)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=6)
    viewImage(mask)
    return mask


def get_seg(mask,img):
    # mask：掩码图
    # img:需要分割的图
    # 复制图像作为分割显示
    draw_img = img.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_num = 1  # 组件的数量
    path = './text/' #保存轮廓信息的路径
    for i, contour in enumerate(contours):
        ares = cv2.contourArea(contour)
        rect = cv2.minAreaRect(contour)
        #拼接图的筛选
        if ares < 100000 or rect[1][0] / rect[1][1] >0.5:  # 过滤面积小于50000的形状
            continue
        #计算轮廓中心点坐标
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x, y, w, h = cv2.boundingRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(draw_img, [box], 0, (0, 255, 0), 2)
        cv2.putText(draw_img, "#{}".format(ares), (x + int(w / 2), y + int(h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 255), 2)
        big_num += 1
    # viewImage(draw_img)
    print(big_num)

    return draw_img

def get_label_single_moduleimg_show(mask,img):
    # 显示并筛选出所有标记的组件
    grat_img = rgb2gray(img)
    label_im = label(mask)
    regions = regionprops(label_im)
    imshow(label_im)
    # --------------------显示区域数据--------------------
    # properties = ['area', 'convex_area', 'bbox_area', 'extent', 'mean_intensity',
    #               'solidity', 'eccentricity', 'orientation']
    # pd.DataFrame(regionprops_table(label_im, grat_img, properties=properties))
    #---------------------筛选-----------------------------------
    masks = []
    bbox = []
    list_of_index = []
    list_of_region = []
    for num, x in enumerate(regions):
        area = x.area  # 区域像素数量
        convex_area = x.convex_area
        if (num != 0 and (area > 100000)):
            print(convex_area / area)  # 大部分大于一 1.01~ 1.04
            masks.append(regions[num].convex_image)
            bbox.append(regions[num].bbox)
            list_of_index.append(num)
            list_of_region.append(x)
    count = len(masks)
    print(count)

    # 画出所划分的轮廓部分
    fig, ax = plt.subplots(2, int(count / 2), figsize=(15, 8))
    for axis, box, mask in zip(ax.flatten(), bbox, masks):
        red = img[:, :, 0][box[0]:box[2], box[1]:box[3]] * mask
        green = img[:, :, 1][box[0]:box[2], box[1]:box[3]] * mask
        blue = img[:, :, 2][box[0]:box[2], box[1]:box[3]] * mask
        image = np.dstack([red, green, blue])
        axis.imshow(image)
    plt.tight_layout()
    plt.show()
    # plt.savefig('fig.png', bbox_inches='tight')  # 替换 plt.show() 保存文件
    #去除背景
    rgb_mask = np.zeros_like(label_im)
    for x in list_of_index:
        rgb_mask += (label_im == x + 1).astype(int)
    red = img[:, :, 0] * rgb_mask
    green = img[:, :, 1] * rgb_mask
    blue = img[:, :, 2] * rgb_mask
    image = np.dstack([red, green, blue])
    imshow(image)

    plt.show()
# 替换 plt.show() 保存文件
    return list_of_region

def get_label_single_module_index(mask,img,taget_pixel):
    #根据目标像素坐标的值 求得所在组串的编号值 并显示所在的组串
    label_im = label(mask)
    regions = regionprops(label_im)
    for num, x in enumerate(regions):
        if x.area < 100000:
            continue
        bbox = x.bbox
        if taget_pixel[0] > bbox[1] and taget_pixel[0] < bbox[3] and taget_pixel[1] > bbox[0] and taget_pixel[1] < bbox[2]:
            #显示出目标像素点所在的组串
            rgb_mask = (label_im == num + 1).astype(int)
            red = img[:, :, 0] * rgb_mask
            green = img[:, :, 1] * rgb_mask
            blue = img[:, :, 2] * rgb_mask
            image = np.dstack([red, green, blue])
            imshow(image)
            plt.show()
            return num
if __name__ == "__main__" :
    sigle_img_path = './images/DJI_20210803111219_0313_W.JPG'
    mosaic_img_path = './result/output_crop_raster_0313.tif'
    sigle_img = cv2.imread(sigle_img_path)
    mosaic_img = cv2.imread(mosaic_img_path)
    # sigle_mask = get_mask(sigle_img, hsv_range=[55, 135, 45, 255, 70, 255])
    #差别主要在于亮度值v
    mosaic_mask = get_mask(mosaic_img, hsv_range=[55, 135, 45, 255, 90, 255])
    #自适应获取阈值得到mask图像
    # mosaic_mask = get_HSV(mosaic_img)
    mosaic_seg_img = get_seg(mosaic_mask,mosaic_img)
    # 通过regionprops函数标注分割区域
    # mosaic_list_of_region = get_label_single_moduleimg_show(mosaic_mask,mosaic_img)

    #目标像素点
    target_pixel = [500,101]
    # 求出像素点所在的组串位置
    module_num = get_label_single_module_index(mosaic_mask,mosaic_img,target_pixel)
    print('缺陷点所在组串编号为：',module_num)




