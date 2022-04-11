import numpy as np
import math as m
import cv2

import test_parper_get_ervery_latlon as xmp



def computeR(Pitch, Yaw, Roll):

    a = Pitch * np.pi / 180
    b = Yaw * np.pi / 180
    g = Roll * np.pi / 180
    # Compute R matrix according to source.
    Rz = np.array(([m.cos(a), m.sin(a), 0],
                   [-1 * m.sin(a), m.cos(a), 0],
                   [0, 0, 1]))

    Ry = np.array(([m.cos(b), 0, -1 * m.sin(b)],
                   [0, 1, 0],
                   [m.sin(b), 0, m.cos(b)]))

    Rx = np.array(([1, 0, 0],
                   [0, m.cos(g), m.sin(g)],
                   [0, -1 * m.sin(g), m.cos(g)]))
    Ryx = np.dot(Rx, Ry)
    R = np.dot(Rz, Ryx)
    R[0, 0] = 1
    R[2, 2] = 1
    return R

def Geo_Correcition(img,Pitch, Yaw, Roll,f):
    Dx = 0.0063
    Dy = 0.0047
    H, W = img.shape[:2]
    u0 = W / 2
    v0 = H / 2

    tmp1 = np.array(([1 / Dx, 0, u0],
                     [0, 1 / Dy, v0],
                     [0, 0, 1]))

    tmp2 = np.array(([f, 0, 0],
                     [0, f, 0],
                     [0, 0, 1]))

    # 报错信息
    try:
        K1 = np.dot(tmp1, tmp2)  # 3*3
        # K1的逆矩阵
        InvK1 = np.linalg.inv(K1)
    except:
        print("矩阵不存在逆矩阵")

    R = computeR(Pitch, Yaw, Roll)
    tmp3 = np.dot(K1, R)
    tmp4 = np.dot(tmp3, InvK1)

    # 像素重映射
    mapx = np.zeros(img.shape[:2], dtype=np.float32)
    mapy = np.zeros(img.shape[:2], dtype=np.float32)
    print('Geo_Correcition start...')
    for i in range(H):
        for j in range(W):
            arr = np.array([j, i, 1])
            dst = np.dot(tmp4, arr.T)
            mapx[i, j] = np.float32(dst[0])
            mapy[i, j] = np.float32(dst[1])
    print('Geo_Correcition over...')
    newimg = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)  # cv2.remap  像素重映射 双三样条插值
    # cv2.imwrite('res223R.jpg', newimg)
    return newimg

if __name__=="__main__":
    img2_path = './images/DJI_20210803111219_0313_W.JPG'
    img2  = cv2.imread(img2_path)
    img_xmp = xmp.get_xmp_info(img2_path)
    Pitch = 0
    Yaw = 180 - img_xmp['Flight-Yaw']
    Roll = 0

    res = Geo_Correcition(img2,Pitch, Yaw, Roll,img_xmp['FocalLength'])
    cv2.imwrite('dji_0313_geo_croo.jpg',res)




