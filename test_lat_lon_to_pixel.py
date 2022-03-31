from osgeo import osr,gdal

infile = "./images/MOSAIC_YC.tif'" #File Path goes here
#Lat Long value
long = 102.30131411
lat =  38.25965909

indataset = gdal.Open(infile)
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
srs = osr.SpatialReference()
srs.ImportFromWkt(wgs84_wkt)

srsLatLong = srs.CloneGeogCS()
ct = osr.CoordinateTransformation(srsLatLong, srs)
(X, Y, height) = ct.TransformPoint(long, lat)

# Report results
print('longitude: %f\t\tlatitude: %f' % (long, lat))
print('X: %f\t\tY: %f' % (X, Y))
#VALUE OF COORDINATE IN METERS
print (X ,Y)

driver = gdal.GetDriverByName('GTiff')
band = indataset.GetRasterBand(1)

cols = indataset.RasterXSize
rows = indataset.RasterYSize

transform = indataset.GetGeoTransform()

xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = -transform[5]
data = band.ReadAsArray(0, 0, cols, rows)

points_list = [(X,Y)] #list of X,Y coordinates

for point in points_list:
    col = int((point[0] - xOrigin) / pixelWidth )
    row = int((yOrigin - point[1] ) / pixelHeight)
    #ROW AND COLUMN VALUE
    print(row,col )
    #Data AT THAT ROW COLUMN
    value = data[row][col]
    print(value)

# if __name__ == "__main__":
#     img1_path = './images/MOSAIC_YC.tif'
#     coors = coords2pixels(img1_path,102.30101630, 38.25953050)
#     print(coors)
