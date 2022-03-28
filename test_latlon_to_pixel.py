#将获取到的经纬度坐标转换成对应的像素坐标
import math
# 由于转换算法适用于以弧度表示的经纬度值，将度数转换为弧度
def degreesToRadians(degrees):
    return (degrees * math.pi) / 180

# latitude 纬度 longitude 经度
def latLonToOffsets(latitude, longitude, mapWidth, mapHeight):
    FE = 180
    radius = mapWidth / (2 * math.pi)
    latRad = degreesToRadians(latitude)
    lonRad = degreesToRadians(longitude + FE)
    x = lonRad * radius
    yFromEquator = radius * math.log(math.tan(math.pi / 4 + latRad / 2))
    y = mapHeight / 2 - yFromEquator

    return x,y

if __name__ == "__main__":
    x,y = latLonToOffsets(38.27164070,102.29611872,4056,3040)
    print(x,y)