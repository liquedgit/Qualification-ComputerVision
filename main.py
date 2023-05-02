import cv2

import ImageProcess
from ImageProcess import *

img = cv2.imread('assets/lena.jpg')

cv2.imshow('Grayscale', grayscale_filter(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Blur', blur_filter(img))
cv2.waitKey(0)
cv2.destroyAllWindows()
