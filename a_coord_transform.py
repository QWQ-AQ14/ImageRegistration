import cv2
from osgeo import gdal,osr
from exif import Image


def lonlat2pixel(ds,lon,lat):
    # 创建目标空间参考
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srsLatLong = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srsLatLong, srs)
    gt = ds.GetGeoTransform()
    # x = 263853.4085 y = 4238056.654 通过经纬度获取地理空间坐标
    # GDA3.0以后TransformPoint的参数为（lat, lon）而不是（lon, lat）
    (X, Y, height) = ct.TransformPoint(lat, lon)
    inv_geometrix = gdal.InvGeoTransform(gt)
    # 获得像素值坐标
    x = int(inv_geometrix[0] + inv_geometrix[1] * X + inv_geometrix[2] * Y)
    y = int(inv_geometrix[3] + inv_geometrix[4] * X + inv_geometrix[5] * Y)

    return (x,y)

#像素值转经纬度
def pixel2latlon(ds,x,y):
    """
       Returns latitude/longitude coordinates from pixel x, y coords

       Keyword Args:
         img_path: Text, path to tif image
         x: Pixel x coordinates. For example, if numpy array, this is the column index
         y: Pixel y coordinates. For example, if numpy array, this is the row index
       """

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
    xoff, a, b, yoff, d, e = gt
    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff

    lat_lon = transform.TransformPoint(xp, yp)

    lat = lat_lon[0]
    lon = lat_lon[1]

    return (lat,lon)

def pixel2geocoord(ds,pixel_x,pixel_y):
    gt = ds.GetGeoTransform()
    #gt[1] :水平空间分辨率 gt[5] :垂直空间分辨率 pixel size
    # 像素坐标转换为投影坐标系projection coordinates
    # col = 996.5
    # row = 782
    geo_x = (pixel_x * gt[1]) + gt[0]
    geo_y = (pixel_y * gt[5]) + gt[3]
    return (geo_x,geo_y)
def center_point_four(ds,center_x,center_y,x_offset,y_offset):
    #offset 需要裁剪的偏移量
    # 输入像素点中心坐标
    upper_left_x = center_x - x_offset
    upper_left_y = center_y - y_offset
    lower_right_x = center_x + x_offset
    lower_right_y = center_y + y_offset
    (upper_left_geo_x, upper_left_geo_y) = pixel2geocoord(ds,upper_left_x,upper_left_y)
    (lower_right_geo_x, lower_right_geo_y) = pixel2geocoord(ds, lower_right_x, lower_right_y)
    return (upper_left_geo_x, upper_left_geo_y,lower_right_geo_x, lower_right_geo_y)

#将经纬度信息转换为小数点形式
def decimal_coords(coords, ref):
 decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
 if ref == "S" or ref == "W":
     decimal_degrees = -decimal_degrees
 return decimal_degrees

# 读取图片
def image_coordinates(img_path):
    with open(img_path, 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            img.gps_longitude
            coords = (decimal_coords(img.gps_latitude,
                      img.gps_latitude_ref),
                      decimal_coords(img.gps_longitude,
                      img.gps_longitude_ref))
        except AttributeError:
            print( 'No Coordinates')
    else:
        print( 'The Image has no EXIF information')
    print(f"Image {src.name}, OS Version:{img.get('software', 'Not Known')} ------")
    print(f"Was taken: {img.datetime_original}, and has coordinates:{coords}")
    return coords

if __name__ == "__main__":
    # 整图
    img1_path = './images/GS.tif'
    # 单幅图
    img2_path = './images/DJI_20210803105452_0067_W.JPG'
    img2=cv2.imread(img2_path)
    # Open tif file
    ds = gdal.Open(img1_path)
    #获取单幅图的经纬度信息
    lat_lon = image_coordinates(img2_path)
    lon,lat = pixel2latlon(ds,996.5,782)
    # lat_lon = [40.49348002,95.70736002]
    x,y = lonlat2pixel(ds,lat_lon[1],lat_lon[0])
    # 中心点坐标 x = 263853.4085 y = 4238056.654
    (upper_left_geo_x, upper_left_geo_y, lower_right_geo_x, lower_right_geo_y) = center_point_four(ds,x,y,img2.shape[1]/2,img2.shape[0]/2)
    window =(upper_left_geo_x, upper_left_geo_y, lower_right_geo_x, lower_right_geo_y)
    #根据地理空间坐标裁剪出对应区域 截出来的图与单幅图有180度旋转偏差
    gdal.Translate('./result/output_crop_raster_0067.tif', './images/GS.tif', projWin=window)
    print(lat_lon)