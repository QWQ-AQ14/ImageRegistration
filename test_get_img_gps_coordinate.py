# 获取单幅图像的经纬度坐标信息
from exif import Image
import PIL
import cv2
import numpy as np

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
    # img_path = 'images/DJI_20210803111219_0313_W.jpg'
    img_path = 'images/DJI_20210803111145_0305_W.JPG'
    image = cv2.imread(img_path)
    image = np.asarray(image, dtype=np.float64)
    # 获取单幅图的经纬度信息
    gps_coords = image_coordinates(img_path)