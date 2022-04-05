# 根据整图中的信息 考虑两点之间的距离 得出经纬度信息
# 判断方法是否可行
lat1 = 40.493028305555555
lon1 = 95.70834169444444
pixel_x_1 = 21842.8904
pixel_y_1 = 16208.8288

pixel_x_2 = 21497.886
pixel_y_2= 16080.8127

pixel_size_x = 0.019576
pixel_size_y = 0.018409
x_long = pixel_size_x * abs(pixel_x_2 - pixel_x_1)
y_long = pixel_size_y * abs(pixel_y_2 - pixel_y_1)

# 纬度
lat = lat1 + (y_long / 30.9/3600)
    # 经度
lon = lon1 - x_long / 23.6/3600
print(lat,lon)