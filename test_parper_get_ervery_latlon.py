import cv2
import test_get_img_gps_coordinate as get_gps
import math
import numpy as np
import PIL.Image
import PIL.ExifTags
import pyexiv2

#将经纬度信息转换为小数点形式
def decimal_coords(coords, ref):
 decimal_degrees = float(coords[0]) + float(coords[1] / 60) + float(coords[2] / 3600)
 if ref == "S" or ref == "W":
     decimal_degrees = -decimal_degrees
 return decimal_degrees

def to_float(str):
    # 去掉前面的符号 并转换为浮点数
    str = str[1:len(str)]
    str = float(str)
    return str

def get_xmp_info(img_path):
    with pyexiv2.Image(img_path) as img:
        data = img.read_exif()
    img = pyexiv2.Image(img_path)
    data = img.read_xmp()
    exif = img.read_exif()
    longitude = data['Xmp.drone-dji.GpsLongitude']
    latitude = data['Xmp.drone-dji.GpsLatitude']

    RelativeAltitude = data['Xmp.drone-dji.RelativeAltitude']
    roll = data['Xmp.drone-dji.FlightRollDegree']
    yaw = data['Xmp.drone-dji.FlightYawDegree']
    pitch = data['Xmp.drone-dji.FlightPitchDegree']

    img_widh = exif['Exif.Photo.PixelXDimension']
    img_height = exif['Exif.Photo.PixelYDimension']

    focal_length = exif['Exif.Photo.FocalLength'].split('/')
    focal_length = float(focal_length[0]) / float(focal_length[1])
    dist = {'Longitude': to_float(longitude), 'Latitude': to_float(latitude), 'Altitude': to_float(RelativeAltitude),
                    'Flight-Roll': float(roll), 'Flight-Pitch': float(pitch), 'Flight-Yaw': float(yaw),
                    'FocalLength': focal_length,'ImageWidth': float(img_widh),'ImageHeight': float(img_height)
                  }
    return dist

if __name__ == "__main__":
    #读取两张图像
    img1_path = './images/DJI_20210803111304_0324_T.JPG'
    img2_path = './images/DJI_20210803111219_0313_W.JPG'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 获取图像中xmp信息
    img1_xmp = get_xmp_info(img1_path)
    img2_xmp = get_xmp_info(img2_path)
    #获取经纬度信息 latitude longitude 纬度40 经度95
    coords1 = (img1_xmp['Latitude'],img1_xmp['Longitude'])
    coords2 = (img2_xmp['Latitude'],img2_xmp['Longitude'])
    print('coords1:',coords1)
    print('coords2:',coords2)
    #每个像素对应的地面距离
    #红外传感器宽度 像元间距 * 图像宽度 单位是mm
    sendor_w_ir = 12  * img1_xmp['ImageWidth']/ 1000
    #可见光传感器宽度 8.8 x 6.6mm
    # The camera sensor is 1/2.3 inch and image size is 6.3116*4.7492mm.
    sendor_w_vis = 6.3116
    sendor_h_vis = 4.7492
    h = img2_xmp['Altitude']
    #地面分辨率 m/pixel
    # gsd_ir = (sendor_w_ir * h ) / (float(img1_xmp['FocalLength']) * float(img1_xmp['ExifImageWidth']))
    gsd_vis_w = (sendor_w_vis * h ) / (float(img2_xmp['FocalLength']) * float(img2_xmp['ImageWidth']))
    gsd_vis_h = (sendor_h_vis * h) / (float(img2_xmp['FocalLength']) * float(img2_xmp['ImageHeight']))


    #中心点坐标
    center_point = np.array([img2.shape[1]/2,img2.shape[0]/2])
    #左上角坐标
    left_point = np.array([1036.3905,852.8012])

    #像素差 距离中心点的实际距离

    sub = center_point - left_point
    sub = sub * np.array([gsd_vis_w,gsd_vis_h])
    offset_lat = 9.791212210578413e-06
    offset_lon = 1.1660749935330442e-05
    # 纬度
    lat = coords2[0] - sub[1] *offset_lat
    # 经度
    lon = coords2[1] + sub[0] * offset_lon
    # # 纬度
    # lat = coords1[0] + (sub[1] / 30.9/3600)
    # # 经度
    # lon = coords1[1] + sub[0] / 23.6/3600
    print(lat,lon)
    target_lat = 40.49333256
    target_lon = 95.70826548
    print(lat - coords2[0],lon - coords2[1])
    print(lat - target_lat,lon - target_lon)
