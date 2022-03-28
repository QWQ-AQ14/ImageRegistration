import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
img1 = mpimg.imread(r"E:\xlq\学术文件\光谱测试数据\第二组数据\支路5.bmp")
img2 = mpimg.imread(r"E:\xlq\学术文件\光谱测试数据\第一组数据\支路1-1.5-1.6μm.bmp")
(a,b,c,d) = cv2.split(img1)
H = np.array([[1.504417669,0.021413131,46.59344852], [-0.009973241,1.465893742,144.3573662],[0,0,1]], np.float32)
image_output = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
res = cv2.merge([a,b,c,d,image_output])
cv2.imwrite("Merged_5C.tif",res)