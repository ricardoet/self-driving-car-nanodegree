import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#read image and convert to gray
image = mpimg.imread('../../../Resources/Images/test.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#gaussian smoothing/blurring
kernel_size = 3 #should be odd number
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

#canny edge
low_threshold = 70
high_threshold = 180
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

#display image and wait for keyclick
plt.imshow(edges, cmap='Greys_r')
plt.waitforbuttonpress()