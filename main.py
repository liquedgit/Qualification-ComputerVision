import cv2

import ImageProcess
from ImageProcess import *
from ShapeDetect import shape_detect

# img = cv2.imread('assets/lena.jpg')
shape = cv2.imread('./assets/shapes.png')
shapeDetect = shape_detect(shape)

cv2.imshow('Shape detected', shapeDetect)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow('Grayscale', grayscale_filter(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('Blur', blur_filter(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('threshold', threshold_binary(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
