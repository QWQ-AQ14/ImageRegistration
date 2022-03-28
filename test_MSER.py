import cv2

img = cv2.imread("images/DJI_20210803111145_0305_W.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mser = cv2.MSER(_delta = 1, _min_area=1)
regions = mser.detect(gray, None)

clone = img.copy()
for region in regions:
    # fit a bounding box to the contour
    (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 1)
cv2.imshow('img', clone)
cv2.waitKey(0)
cv2.destroyAllWindows()