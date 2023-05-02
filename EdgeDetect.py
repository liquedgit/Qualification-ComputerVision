import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('assets/zek.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 150, 250)

plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.title('grayscale image')
plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')


plt.show()


