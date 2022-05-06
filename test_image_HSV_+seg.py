from skimage.color import rgb2hsv,rgb2gray
import cv2
import numpy as np
from skimage.filters import threshold_otsu
#convert to hsv scale
import matplotlib.pyplot as plt
from a_module_matching import get_region
def hsv_range(sample,mask):
    #生成黑白mask图像
    # res = np.ones_like(sample)
    # res[mask == True] = (255, 255, 255)
    # get the desired mask and show in original image
    red = sample[:, :, 0] * mask
    green = sample[:, :, 1] * mask
    blue = sample[:, :, 2] * mask
    mask = np.dstack((red, green, blue))
    return mask
img2_path = './result/output_crop_raster_0313.tif'
sample = cv2.imread(img2_path)
sample_h= rgb2hsv(sample)
sample_g = rgb2gray(sample)
Hue_threshold = threshold_otsu(sample_h[:,:,0])
s_threshold = threshold_otsu(sample_h[:,:,1])
v_threshold = threshold_otsu(sample_h[:,:,2])
sample_hbw = sample_g < Hue_threshold
sample_sbw = sample_g < s_threshold
sample_vbw = sample_g < v_threshold
#为了去除拼接图中组串的阴影部分
sample_sbw_v = sample_g > s_threshold
otsu_mask = sample_vbw * sample_sbw_v
otsu_mask_h = hsv_range(sample,sample_hbw)
otsu_mask_s = hsv_range(sample,sample_sbw)
otsu_mask_v = hsv_range(sample,sample_vbw)
otsu_mask_s_v = hsv_range(sample,otsu_mask)
# 提取轮廓
s_region = get_region(sample_vbw)
#展示对三个分量分别进行OTSU阈值化的效果
# fig, ax = plt.subplots(1, 4, figsize=(15,5))
# ax[0].imshow(otsu_mask_h)
# ax[0].set_title('OTSU H',fontsize=15);
# ax[1].imshow(otsu_mask_s)
# ax[1].set_title('OTSU S',fontsize=15);
# ax[2].imshow(otsu_mask_v)
# ax[2].set_title('OTSU V',fontsize=15);
# ax[3].imshow(otsu_mask_s_v)
# ax[3].set_title('sample_sbw',fontsize=15);

# cv2.imwrite('./result/mosaic0313_S_V.png',otsu_mask_s_v)


