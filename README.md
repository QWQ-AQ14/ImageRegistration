# 1.如何实现整图中经纬度坐标与像素坐标的实现？

 - [x] 已解决

# 2.如何读取大容量整图并截取子图？

 - [x] 通过像素坐标划定一定的范围，然后转换成地理空间坐标进行裁剪
 - [x] [GDAL裁剪子图的方法](https://stackoverflow.com/questions/38242716/how-to-crop-a-raster-image-by-coordinates-in-python)
# 如何通过读取大容量tif图?
- [重新安装conda环境](https://gis.stackexchange.com/questions/291921/cannot-install-a-version-of-gdal-via-anaconda-that-permits-reading-bigtiffs)
![在这里插入图片描述](https://img-blog.csdnimg.cn/c7030b0b91a04eee8852902df6caa9f6.png)

# 如何通过中心点坐标求得其他像素点的经纬度信息？
## 计算无人机的GSD
GSD：一个像素对应的地面距离
如何计算ZENMUSE H20T的传感器尺寸
[How to calculate the flight height based on GSD?](https://preolix.com/en/aprende-a-calcular-la-altura-de-vuelo-para-inspeccion-de-plantas-solares-con-drones/)

可见光传感器`sensor size 6.3116*4.7492mm`
红外传感器尺寸 =  像元间距 * 图像宽度

![在这里插入图片描述](https://img-blog.csdnimg.cn/6b575d99a3504785bbb52462e2441543.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAQVExNF8=,size_20,color_FFFFFF,t_70,g_se,x_16)
## 如何获得单幅图的相对高度
[DJI 精灵4 飞行配置](https://www.dji.com/cn/phantom-4-rtk/info)

- 可根据公式 `H=36.5*GSD` 大致确定合适的飞行高度，考虑地形的起伏，建议设置的飞行高度不大于计算出的 H，其中 GSD 单位为厘米，H 单位为米。如 GSD2.74 cm 对应飞行高度约100米
- 精灵 4 RTK 拍摄的照片中会同时记录海拔高度和相对于返航点的相对高度。建图时读取的是基于对应椭球的大地高（椭球高）（字段：`AbsoluteAltitude`）。相对于返航点的高度可在 XMP 文件中读取（字段：`RelativeAltitude`）
- [利用pyexiv2库读取图片中的XMP信息](https://github.com/LeoHsiao1/pyexiv2/blob/master/docs/Tutorial-cn.md)
![在这里插入图片描述](https://img-blog.csdnimg.cn/ba3fabc2cb36438aaeb83c2133c40abd.png)
