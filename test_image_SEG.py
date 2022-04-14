# 分割组件进行编号
import cv2
import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

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
    viewImage(mask)
    #不腐蚀之前 可分割小组件
    kernel = np.ones((3, 3), np.uint8)
    #腐蚀 是缩小
    mask = cv2.erode(mask, kernel, 2)
    viewImage(mask)
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
        x, y, w, h = cv2.boundingRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(draw_img, [box], 0, (0, 255, 0), 2)
        cv2.putText(draw_img, "#{}".format(ares), (x + int(w / 2), y + int(h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 255), 2)
        coords = np.array2string(contour)
        # 将轮廓信息写入文件夹
        open(path + 'contour_%d.txt' % big_num, "w").write(coords)
        big_num += 1
    viewImage(draw_img)
    print(big_num)

    return draw_img


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
