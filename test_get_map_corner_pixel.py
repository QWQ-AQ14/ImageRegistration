#!/usr/bin/env python3
import sys
import math
import gdal
import osr


def main():
    method = sys.argv[1]
    file = sys.argv[2]
    # raise Exception(file)
    src = gdal.Open(file)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    sizeX = src.RasterXSize
    sizeY = src.RasterYSize

    prj = src.GetProjection()
    srs = osr.SpatialReference(wkt=prj)

    if method == 'bounds':
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(srs, target)
        lrx = ulx + (sizeX * xres)
        lry = uly + (sizeY * yres)
        ulx, uly, elv = transform.TransformPoint(ulx, uly)
        lrx, lry, elv = transform.TransformPoint(lrx, lry)
        # ulx, uly is the upper left corner, lrx, lry is the lower right corner
        print("[%s, %s, %s, %s]" % (ulx, uly, lrx, lry))
    elif method == 'bounds-geojson-format':
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(srs, target)
        lrx = ulx + (sizeX * xres)
        lry = uly + (sizeY * yres)
        ulx, uly, elv = transform.TransformPoint(ulx, uly)
        lrx, lry, elv = transform.TransformPoint(lrx, lry)

        print("[[[%s,%s],[%s,%s],[%s,%s],[%s,%s],[%s,%s]]]" % (ulx, uly, ulx, lry, lrx, lry, lrx, uly, ulx, uly))
    elif method == 'size':
        print("%s %s" % (sizeX, sizeY))
    elif method == 'res':
        print("%s %s" % (xres, yres))
    elif method == 'newres':
        if srs.IsProjected:
            if (srs.GetAttrValue('projcs') != 'WGS 84 / Pseudo-Mercator'):
                print(
                    "\033[31mInput file better to be EPSG:3857 WGS 84 / Pseudo-Mercator, but EPSG:4326 WGS 84 is fine\033[0m",
                    file=sys.stderr)
        else:
            print("\033[31mERROR", file=sys.stderr)
            exit(1)
        '''
        Original
        Size is 3253, 2224
        Block=3253x2
        res 465
        south-sudan-landuse-newres-46-colored.tif
        Size is 32558, 22259
        res 46.5

        south-sudan-landuse-newres-40-colored.tif
        Size is 37848, 25876
        Block=37848x1s
        south-sudan-landuse-newres-12-colored
        Size is 126161, 86253
         Block=126161x1
        south-sudan-landuse-newres-20-colored.tif
        Size is 75697, 51752
         Block=75697x1
        '''
        AIM_SIZEX = 32558
        ratio = AIM_SIZEX / sizeX
        if (srs.GetAttrValue('projcs') == 'WGS 84 / Pseudo-Mercator' and ratio < 1):
            newxres = xres / ratio
            newyres = abs(yres) / ratio
            print("%s %s" % (newxres, newyres))
        else:
            digits = (len(str(math.ceil(ratio))) - 1)

            # magificant of the ratio/// for easy calc purpose
            mag = 10 ** digits

            # detect the second significant value of the numbers
            secondSign = 0
            if str(ratio)[1] != '.':
                secondSign = int(str(ratio)[1])
            if secondSign >= 5:
                mag = mag * 1.5

            newxres = xres / mag
            newyres = abs(yres) / mag

            print("%s %s" % (newxres, newyres))
    elif method == 'projection':
        if srs.IsProjected:
            print(srs.GetAttrValue('projcs'))
        else:
            print("\033[31mERROR", file=sys.stderr)
            exit(1)
    elif method == 'max-value':
        if len(sys.argv) > 3 and len(sys.argv[3]) > 0:
            md = src.GetMetadata_Dict()
            print(md[sys.argv[3] + "#vmax"])
        else:
            srcband = src.GetRasterBand(1)
            stats = srcband.GetStatistics(True, True)
            print(stats[1])
    elif method == 'min-value':
        if len(sys.argv) > 3 and len(sys.argv[3]) > 0:
            md = src.GetMetadata_Dict()
            print(md[sys.argv[3] + "#vmin"])
        else:
            srcband = src.GetRasterBand(1)
            stats = srcband.GetStatistics(True, True)
            print(stats[0])

    src = None


usage = '''
USAGE:
    main.py size "tiff"
    main.py bounds "tiff"
    main.py bounds-geojson-format "tiff"
    main.py res "tiff"
    main.py newres "tiff is better to be EPSG:3857 standard"
    main.py projection "tiff"
'''
if __name__ == '__main__':
    num_args = len(sys.argv)
    if num_args < 3:
        print(usage)
        exit()
    main()