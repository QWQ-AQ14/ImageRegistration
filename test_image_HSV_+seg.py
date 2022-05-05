from skimage.color import rgb2hsv
import cv2
import numpy as np
#convert to hsv scale
import matplotlib.pyplot as plt
img2_path = './images/DJI_20210803111020_0285_W.JPG'
sample = cv2.imread(img2_path)
sample_h= rgb2hsv(sample)
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
red = sample[:,:,0]*mask
green = sample[:,:,1]*mask
blue = sample[:,:,2]*mask
mask2 = np.dstack((red,green,blue))
ax[1].imshow(mask)
ax[2].imshow(mask2)
ax[1].set_title('Mask',fontsize=15)
ax[2].set_title('Final Image',fontsize=15)
plt.tight_layout()
plt.show()