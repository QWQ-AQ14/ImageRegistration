import cv2
import test_get_img_gps_coordinate as get_gps
import math
import numpy as np
if __name__ == "__main__":
    #读取两张图像
    img1_path = './images/DJI_20210803111304_0324_T.JPG'
    img2_path = './images/DJI_20210803111305_0324_W.JPG'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    #获取经纬度信息 latitude longitude 纬度40 经度95
    img1_exif,coords1 = get_gps.image_coordinates(img1_path)
    img2_exif,coords2 = get_gps.image_coordinates(img2_path)

    #每个像素对应的地面距离
    #传感器宽度 像元间距 * 图像宽度 单位是mm
    sendor_w = 12  * img1_exif.pixel_x_dimension / 1000
    h = 65 #相对高度 单位m
    #地面分辨率 m/pixel
    gsd = (sendor_w * h * 100) / (img1_exif.focal_length * img1_exif.pixel_x_dimension)
    print(gsd)

    #中心点坐标
    center_point = np.array([img1.shape[1]/2,img1.shape[0]/2])
    #左上角坐标
    left_point = np.array([0,0])

    #像素差
    sub = center_point - left_point
    sub = sub * np.array([gsd,gsd])

    # 纬度
    lat = coords1[0] + (sub[1] / 30.9/3600)
    # 经度
    lon = coords1[1] + sub[0] / 23.6/3600
    print(lat,lon)
    print(lat - coords1[0],lon - coords1[1])
