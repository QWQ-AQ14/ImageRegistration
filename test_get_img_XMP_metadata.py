import pyexiv2
import os
from tabulate import tabulate

def convert_fractions_to_float(fraction):
    return fraction.numerator / fraction.denominator


def convert_dms_to_deg(dms):
    d = convert_fractions_to_float(dms[0])
    m = convert_fractions_to_float(dms[1]) / 60
    s = convert_fractions_to_float(dms[2]) / 3600
    deg = d + m + s
    return deg

def to_float(str):
    # 去掉前面的符号 并转换为浮点数
    str = str[1:len(str)]
    str = float(str)
    return str

if __name__ == "__main__":
    img_path = './images/DJI_20210803111145_0305_W.JPG'
    # 通过 with 关键字打开图片时，它会自动关闭图片以释放用于存储图片数据的内存
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

    print(tabulate([['Longitude', to_float(longitude)], ['Latitude', to_float(latitude)], ['Altitude', to_float(RelativeAltitude)],
                    ['Flight-Roll', float(roll)], ['Flight-Pitch', float(pitch)], ['Flight-Yaw', float(yaw)],
                    ['FocalLength', focal_length],['ImageWidth', float(img_widh)],['ImageHeight', float(img_height)]],
                   headers=["Field", "Value(deg)"],
                   tablefmt='orgtbl',
                   numalign="right"))


