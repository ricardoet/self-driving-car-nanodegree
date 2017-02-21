import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#read image
image = mpimg.imread('../../../Resources/Images/test.jpg')

#get image data and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

#thresholds for color
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

#define vertices for triangular mask
left_bottom = [0, ysize]
right_bottom = [xsize, ysize]
apex = [xsize/2, ysize/2]

#perform linear fit (y=Ax+B) to each of the three sides of the triangle
#np.polyfit returns the coefficients [A,B] of the fit
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

#mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) \
                | (image[:,:,1] < rgb_threshold[1]) \
                | (image[:,:,2] < rgb_threshold[2])

#find the region inside the triangle
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) \
                & (YY > (XX*fit_right[1] + fit_right[1])) \
                & (YY < (XX*fit_bottom[0] + fit_bottom[1]))

#mask color and region
color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
#color pixels red where both selections met
line_image[~color_thresholds & region_thresholds] = [255, 0 ,0]

#display the image and show region and color selections
plt.imshow(image)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.plot(x, y, 'b--', lw=4)
plt.imshow(color_select)
plt.imshow(line_image)

plt.waitforbuttonpress()