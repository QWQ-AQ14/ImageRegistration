import cv2
import test_get_img_gps_coordinate as get_gps
import math
import numpy as np
if __name__ == "__main__":
    #读取两张图像
    img1_path = './images/DJI_20210803111301_0323_W.JPG'
    img2_path = './images/DJI_20210803111305_0324_W.JPG'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    #获取经纬度信息 latitude longitude 纬度40 经度95
    coords1 = get_gps.image_coordinates(img1_path)
    coords2 = get_gps.image_coordinates(img2_path)
    # 通过图像间经纬度夹角计算航线与经纬线的夹角
    angle = math.atan((coords2[1] - coords1[1] ) / (coords2[0] - coords1[0] ))

    # 求无人机的视场范围
    h = 1151.936 #无人机高度
    wide_angle = 82.9

    s = 2 * h * math.tan(wide_angle/2)

    #每个像素对应的地面距离
    l1 = s / img1.shape[1]
    l2 = s / img1.shape[0]
    print(l1)

    #中心点坐标
    center_point = np.array([img1.shape[1]/2,img1.shape[0]/2])
    #左上角坐标
    left_point = np.array([0,0])

    #像素差
    sub = center_point - left_point
    sub = sub * np.array([l1,l2])

    # 纬度
    lat = coords1[0] + sub[1] / 30.9
    # 经度
    lon = coords1[1] + sub[0] / 23.6
    print(lat,lon)





