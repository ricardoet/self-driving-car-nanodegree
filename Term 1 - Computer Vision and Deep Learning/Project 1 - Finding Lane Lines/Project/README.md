#**Project #1: Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


<p align="center">
  <img src="https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/steps/yellow.gif"/>
</p>

## Overview

When we drive, we use our eyes to decide where to go. The lines on the road show us where the lanes are and act as our constant reference for where to steer the vehicle. Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

For this project we were given a Jupyter Notebook to work on. Here's a quick [description](https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python) of how Jupyter Notebooks work. Also, it is important for you to know basics of OpenCV as it's a the most important library we're using on this project.

All the files regarding this project are [here](https://github.com/ricardoet/self-driving-car-nanodegree/tree/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project).

## Goals

The goal for this first project on the Nanodegree is to create a pipelines which allow us to identify (via a red line) the lanes where the car is driving, first on images and then on video. For this, we are prompted to use the knowledge learned mainly via the quizzes but we are open to use any other knowledge to further enhance our pipeline:
* [Color Selection](https://github.com/ricardoet/self-driving-car-nanodegree/tree/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Quizzes/Quiz1%20-%20Color%20Selection)
* [Color Region](https://github.com/ricardoet/self-driving-car-nanodegree/tree/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Quizzes/Quiz2%20-%20Color%20Region)
* [Canny Edges](https://github.com/ricardoet/self-driving-car-nanodegree/tree/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Quizzes/Quiz3%20-%20Canny%20Edges)
* [Hough Transform](https://github.com/ricardoet/self-driving-car-nanodegree/tree/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Quizzes/Quiz4%20-%20Hough%20Transform)

## Pipeline

### 1. Converting image to grayscale
Converting an image to grayscale has several pros. Luminance is far more important at identifying features in images than chrominance. It also makes it way less complex as we work with only one plane instead of the full RGB planes. While I have no doubt we'll be working with full-color images on the future, for this first project we don't need that.

      image = mpimg.imread('test_images/' + imageName)
      grayImage = grayscale(image)

![](https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/steps/original%20to%20grayscale.jpg)

---
### 2. Gaussian blur
[Gaussian blur](http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur) is a filter which smooths our image. It's an important part of the pre-processing stage because it helps reduce the image noise.

It's important to know that an odd-sized filter is needed for the Gaussian blur to work.

      gaussianImage = gaussian_blur(grayImage, 5)

![](https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/steps/gray%20to%20gaussian.png)

---    
### 3. Canny edge conversion
[Canny edge](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html) conversion is the first step on starting to look for the lines. Given a low and high threshold, the algorithm first detects strong edge (strong gradient) above the high threshold and include those. Then pixels between the low and high threshold will be included only if they are part of a line of strong edges (higher than high threshold).

The algorithm returns a binary image with the edges on white and everything else on black.

     cannyImage = canny(gaussianImage, 200, 257)

![](https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/steps/gaussian%20to%20canny.png)

---
### 4. Mask area of interest
Afterwards, I masked the image to include only the lane we are interested in, knowing that our camera is static and always on the same position, this method works well.

Masking was done by trial and error calculating the best polygon that included the lanes I wanted to process.

     ysize = image.shape[0]
     xsize = image.shape[1]
     vertices = np.array([[(140,ysize),(xsize-50, ysize), (xsize/2+20, ysize/2+40), (xsize/2-20, ysize/2+40)]], dtype=np.int32)
     maskedImage = region_of_interest(cannyImage, vertices)

On the actual pipeline I obviously processed the output from canny edges to the masked function, but to illustrate this better the image below is the starting image masked.
![](https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/steps/original%20to%20masked.png)

---
### 5. Hough transform
After masking the Canny Edge output it's time to detect the lines. The [Hough Transform](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html) is a transform used specifically for this: to detect straight lines. This function outputs an array of points that form lines in the form of (x0, y0, x1, y1) which we then use to draw the lines on top of the original image.

     rho = 1 # distance resolution in pixels of the Hough grid
     theta = (np.pi/180)*1  # angular resolution in radians of the Hough grid
     threshold = 50     # minimum number of votes (intersections in Hough grid cell)
     min_line_length = 100  #minimum number of pixels making up a line
     max_line_gap = 160    # maximum gap in pixels between connectable line segments
     houghImage = hough_lines(maskedImage, rho, theta, threshold, min_line_length, max_line_gap)

Using only this will output a bunch of lines, like so:
<p align="center">
<img src="https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/steps/pre-final.jpg" width="400">
</p>

On this part it is important to fine-tune all of the parameters with what works best to draw lines only over the lanes.
So far this works at identifying the lane lines but what we want is to draw only one red line over each of the two lanes. For this we need a little bit of post-processing of the Hough Transform output.

---
### 6. Lines post-processing
This is the final part of the pipeline; where we get the final (x0,y0, x1,y1) arrays from the hough transform and output only two lines. For this, I developed several functions, the main one called "filter_lines". The pipeline for this specific step is the following:

1. Feed pre-calculated slopes (-0.7 and 0.6) and process the whole video once to get the real average of the slopes of the lanes using a big threshold (30%).
2. Having the initial real average slopes we reprocess the video looking for the highest left and right point and filter the average a bit more with a smaller threshold (10%).
3. Now that we have the highest left and right point which hopefully are part of the lanes we extrapolate the lowest to the highest point (so both lines have the same distance).
4. To finish we extrapolate both high points to the lowest part of the image using the average slope for each line. This gives us a really good approximation of the whole line.

Afterwards we output two (x0,y0, x1,y1) arrays (one for each lane) and draw them on top of the original image. The final output looks like this:

<p align="center">
<img src="https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/steps/final.jpg" width="400">
</p>

Here you can see the big difference the post-processing of the Hough Transform makes, leaving only 2 red lines on top of the original image. Afterwards there's a bit more code to implement this on a video (which is just a series of images) and you can find both processed videos here:

1. [Video 1](https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/white.mp4)
2. [Video 2](https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/yellow.mp4)


## Potential Shortcomings
There's a ton of potential shortcomings on my algorithm, no wonder in real life we use DNN and not straightforward programming for this. Here are the top 3 potential shortcomings I found:

### 1. Change in road color
One of the obvious potential shortcomings (as we can see if we try to process the challenge video) is change of color in the road as our fine-tuning won't work because either it's too general and it fails to successfully identify only the lanes or it's too specific and fails to get the lanes on different conditions.

### 2. Difference in lighting day/night
Difference in lighting is another possible shortcoming. For this project we worked with 2 videos recorded in broad daylight. Identifying lanes at night would be much harder because of the smaller contrast (gradient). Canny edge conversion would have to be really fine-tuned and working with the car headlights would definitely be much harder.

### 3. Sharper edges misinterpretation
Getting close to a sharp curve/edge would definitely cause problems on the current algorithm. As we get closer and closer to, for example, a 90° turn, the lenght of the lane detected would get smaller and smaller until the cars gets right before the curve. At this point the algorithm will just stop working and identify lines at the end of the road or something else.



## Possible Improvements
As on the shortcomings, there's naturally a bunch of possible improvements I could make to make this algorithm better at recognizing lane lines. Here's my top 3:

### 1. Smoothing the change between images
As you can see my [video](https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/white.mp4) has much more jitter in comparison to the [expected video](https://github.com/ricardoet/self-driving-car-nanodegree/blob/master/Term%201%20-%20Computer%20Vision%20and%20Deep%20Learning/Project%201%20-%20Finding%20Lane%20Lines/Project/P1_example.mp4).

A possible improvement would be to implement some kind of smoothing/averaging the change in between images so that the video looks way smoother, even though the lane may not be as exact.

### 2. Looking for white/yellow lanes
Another possible improvement is to actually look for white/yellow lanes given that we are sure the lane should be either white or yellow. On the actual algorithm we don't really care about color but looking specifically for this color could be used as a way to eliminate possible false-positives.

### 3. Measure the distance between the two lanes
Other way to eliminate possible false-positives is to measure the distance between both possible lanes. Given that the camera is always on the same position, the lane size shouldn't vary too much. It's important to note that the lane size should always be measure at the same distance from the car for this to work.



