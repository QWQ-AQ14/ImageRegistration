from skimage.color import rgb2hsv,rgb2gray
import cv2
import numpy as np
from skimage.filters import threshold_otsu
#convert to hsv scale
import matplotlib.pyplot as plt

def hsv_range(sample,mask):
    # get the desired mask and show in original image
    red = sample[:, :, 0] * mask
    green = sample[:, :, 1] * mask
    blue = sample[:, :, 2] * mask
    mask = np.dstack((red, green, blue))
    return mask
img2_path = './images/DJI_20210803105452_0067_W.JPG'
sample = cv2.imread(img2_path)
sample_h= rgb2hsv(sample)
sample_g = rgb2gray(sample)
Hue_threshold = threshold_otsu(sample_h[:,:,0])
s_threshold = threshold_otsu(sample_h[:,:,1])
v_threshold = threshold_otsu(sample_h[:,:,2])
sample_hbw = sample_g < Hue_threshold
sample_sbw = sample_g < s_threshold
sample_vbw = sample_g < v_threshold
sample_sbw_v = sample_g < v_threshold + s_threshold
s_mask = sample[:, :, 2] < v_threshold
v_mask = sample[:, :, 2] > v_threshold - 0.25
otsu_mask = s_mask * v_mask
otsu_mask_h = hsv_range(sample,sample_hbw)
otsu_mask_s = hsv_range(sample,sample_sbw)
otsu_mask_v = hsv_range(sample,sample_vbw)
otsu_mask_s_v = hsv_range(sample,sample_sbw_v)
#展示对三个分量分别进行OTSU阈值化的效果
fig, ax = plt.subplots(1, 4, figsize=(15,5))
ax[0].imshow(otsu_mask_h)
ax[0].set_title('OTSU H',fontsize=15);
ax[1].imshow(otsu_mask_s)
ax[1].set_title('OTSU S',fontsize=15);
ax[2].imshow(otsu_mask_v)
ax[2].set_title('OTSU V',fontsize=15);
ax[3].imshow(sample_sbw)
ax[3].set_title('sample_sbw',fontsize=15);
#graph per HSV Channel
fig, ax = plt.subplots(1, 3, figsize=(15,5))
ax[0].imshow(sample_h[:,:,0], cmap='hsv')
ax[0].set_title('Hue',fontsize=15)
ax[1].imshow(sample_h[:,:,1], cmap='hsv')
ax[1].set_title('Saturation',fontsize=15)
ax[2].imshow(sample_h[:,:,2], cmap='hsv')
ax[2].set_title('Value',fontsize=15);

plt.show()
fig, ax = plt.subplots(1,3,figsize=(15,5))
im = ax[0].imshow(sample_h[:,:,0],cmap='hsv')
fig.colorbar(im,ax=ax[0])
ax[0].set_title('Hue Graph',fontsize=15)
#set the lower and upper mask based on hue colorbar value of the desired fruit
lower_mask = sample_h[:,:,0] > 0.05
upper_mask = sample_h[:,:,0] < 0.2
mask = upper_mask*lower_mask
# get the desired mask and show in original image
mask2 = hsv_range(sample,mask)
ax[1].imshow(mask)
ax[2].imshow(mask2)
ax[1].set_title('Mask',fontsize=15)
ax[2].set_title('Final Image',fontsize=15)
plt.tight_layout()
plt.show()