import glob
import os
import pickle
import sys

import gdal
# import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from osgeo import osr
import PIL
from PIL import Image, TiffImagePlugin
from shapely.geometry import Point, Polygon, box



def pixel2coord(img_path, x, y):
    """
    Returns latitude/longitude coordinates from pixel x, y coords

    Keyword Args:
      img_path: Text, path to tif image
      x: Pixel x coordinates. For example, if numpy array, this is the column index
      y: Pixel y coordinates. For example, if numpy array, this is the row index
    """
    # Open tif file
    ds = gdal.Open(img_path)

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

    # GDAL affine transform parameters, According to gdal documentation xoff/yoff are image left corner, a/e are pixel wight/height and b/d is rotation and is zero if image is north up.
    xoff, a, b, yoff, d, e = gt

    xp = a * x + b * y + xoff
    yp = d * x + e * y + yoff

    lat_lon = transform.TransformPoint(xp, yp)

    xp = lat_lon[0]
    yp = lat_lon[1]

    return (xp, yp,transform)

def find_img_coordinates(img_array, image_filename):
    img_coordinates = np.zeros((img_array.shape[0], img_array.shape[1], 2)).tolist()
    for row in range(0, img_array.shape[0]):
        for col in range(0, img_array.shape[1]):
            img_coordinates[row][col] = Point(pixel2coord(img_path=image_filename, x=col, y=row))
    return img_coordinates


# def find_image_pixel_lat_lon_coord(image_filenames, output_filename):
#     """
#     Find latitude, longitude coordinates for each pixel in the image
#
#     Keyword Args:
#       image_filenames: A list of paths to tif images
#       output_filename: A string specifying the output filename of a pickle file to store results
#
#     Returns image_coordinates_dict whose keys are filenames and values are an array of the same shape as the image with each element being the latitude/longitude coordinates.
#     """
#     image_coordinates_dict = {}
#     for image_filename in image_filenames:
#         print('Processing {}'.format(image_filename))
#         img = Image.open(image_filename)
#         img_array = np.array(img)
#         img_coordinates = find_img_coordinates(img_array=img_array, image_filename=image_filename)
#         image_coordinates_dict[image_filename] = img_coordinates
#         with open(os.path.join(DATA_DIR, 'interim', output_filename + '.pkl'), 'wb') as f:
#             pickle.dump(image_coordinates_dict, f)
#     return image_coordinates_dict

if __name__ == "__main__":
    img1_path = './images/MOSAIC_YC.tif'
    xp, yp,transform = pixel2coord(img1_path,1403.0759,1094.4142)
    print(xp, yp)



