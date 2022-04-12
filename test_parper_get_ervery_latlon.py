import cv2
import test_get_img_gps_coordinate as get_gps
import math
import numpy as np
import PIL.Image
import PIL.ExifTags
import pyexiv2
from osgeo import gdal,osr

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
#根据经纬度得出地理空间坐标
def lonlat2pixel(lon,lat):
    img1_path = './images/GS.tif'
    ds = gdal.Open(img1_path)
    # 创建目标空间参考
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srsLatLong = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srsLatLong, srs)
    gt = ds.GetGeoTransform()
    # x = 263853.4085 y = 4238056.654 通过经纬度获取地理空间坐标
    # GDA3.0以后TransformPoint的参数为（lat, lon）而不是（lon, lat）
    (X, Y, height) = ct.TransformPoint(lat, lon)
    return (X, Y)

def geoXY2latlon(xp, yp):
    """
       Returns latitude/longitude coordinates from pixel x, y coords

       Keyword Args:
         img_path: Text, path to tif image
         x: Pixel x coordinates. For example, if numpy array, this is the column index
         y: Pixel y coordinates. For example, if numpy array, this is the row index
       """
    img1_path = './images/GS.tif'
    ds = gdal.Open(img1_path)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    # In this case, we'll use WGS 84
    # This is necessary becuase Planet Imagery is default in UTM (Zone 15). So we want to convert to latitude/longitude
    wgs84_wkt = """
       GEOGCS["WGS 84",
           DATUM["WGS_1984",
               SPHEROID["WGS 84",6378137,298.257223563,
                   AUTHORITY["EPSG","7030"]],
               AUTHORITY["EPSG","6326"]],
           PRIMEM["Greenwich",0,
               AUTHORITY["EPSG","8901"]],
           UNIT["degree",0.01745329251994328,
               AUTHORITY["EPSG","9122"]],
           AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    gt = ds.GetGeoTransform()
    lat_lon = transform.TransformPoint(xp, yp)

    lat = lat_lon[0]
    lon = lat_lon[1]

    return (lat,lon)
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
    #计算图像中心点对应的地理空间坐标系
    (geo_x,geo_y)=lonlat2pixel(coords2[1],coords2[0])

    #每个像素对应的地面距离
    # The camera sensor is 1/2.3 inch and image size is 6.3116*4.7492mm.
    sendor_w_vis = 6.3116
    sendor_h_vis = 4.7492
    h = img2_xmp['Altitude']
    #地面分辨率 m/pixel
    # gsd_ir = (sendor_w_ir * h ) / (float(img1_xmp['FocalLength']) * float(img1_xmp['ExifImageWidth']))
    gsd_vis_w = (sendor_w_vis * h ) / (float(img2_xmp['FocalLength']) * float(img2_xmp['ImageWidth']))
    gsd_vis_h = (sendor_h_vis * h) / (float(img2_xmp['FocalLength']) * float(img2_xmp['ImageHeight']))

    gt = [geo_x, gsd_vis_w, 0,geo_y,0,gsd_vis_h]
    #中心点坐标
    center_point = np.array([img2.shape[1]/2,img2.shape[0]/2])
    #左上角坐标
    left_point = np.array([img2.shape[1],img2.shape[0]])

    #像素差 距离中心点的实际距离
    sub = center_point - left_point
    #获取目标的地理空格坐标系
    target_geo_x = (sub[0] * gt[1]) + gt[0]
    target_geo_y = (sub[1] * gt[5]) + gt[3]

    (lat,lon) = geoXY2latlon(target_geo_x,target_geo_y)
    print(lat,lon)
    target_lat = 40.49333256
    target_lon = 95.70826548
    print(lat - coords2[0],lon - coords2[1])
    print(lat - target_lat,lon - target_lon)
