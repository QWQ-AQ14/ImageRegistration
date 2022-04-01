import gdal
from osgeo import osr

def lonlat2pixel(ds,lon,lat):
    # 获取GDAL仿射矩阵
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srsLatLong = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srsLatLong, srs)
    gt = ds.GetGeoTransform()
    # x = 263853.4085 y = 4238056.654 通过经纬度获取地理空间坐标
    (X, Y, height) = ct.TransformPoint(lon, lat)
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

if __name__ == "__main__":
    img1_path = './images/MOSAIC_YC.tif'
    # Open tif file
    ds = gdal.Open(img1_path)
    lon,lat = pixel2latlon(ds,996.5,782)
    x,y = lonlat2pixel(ds,102.30101606557373,38.259530498231456)
    print(lat,lon)