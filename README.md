# CarND-VehicleDetectionAndTracking-P5
Self-Driving Cars Nanodegree last project: vehicle detection and tracking

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal of this project is to implement a software pipeline in order to detect vehicles in a video. All the process will be described in this documentation which follows the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) from Udacity as starting point.

[//]: # (Image References)
[image1]: ./output_imgs/color_spaces_features.png
[image2]: ./output_imgs/combined_features.png
[image3]: ./output_imgs/dataset_examples.png
[image4]: ./output_imgs/bad_pipeline_res.png
[image5]: ./output_imgs/final_pipeline_test1.png
[image6]: ./output_imgs/final_pipeline_test2.png
[image7]: ./output_imgs/final_pipeline_test3.png
[image8]: ./output_imgs/final_pipeline_test4.png
[image9]: ./output_imgs/final_pipeline_test5.png
[image10]: ./output_imgs/final_pipeline_test6.png
[image11]: ./output_imgs/hist_feat.png
[image12]: ./output_imgs/histogram_feature1.png
[image13]: ./output_imgs/histogram_feature2.png
[image14]: ./output_imgs/hog_features.png
[video1]: ./project_video_output.mp4
[video2]: ./test_video_output.mp4


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

In order to help the reviewer I will save some examples of the output from each stage of the pipeline located in the `output_images`folder. The video called `project_video_output.mp4` is the video of the pipeline execution.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the section 2.1 of the IPython notebook (third code cell).  

I started by reading in all the `vehicle` and `non-vehicle` images (section 1 - Read datasets).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image3]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image14]

The funcion which extracts the spatial features is called `bin_spatial`. Some of the results are the following:

![alt_text][image1]

At the same time, the function which extracts the color histogram is called `color_hist`. Some results are the following:

![alt_text][image12]
![alt_text][image13]

We can notice the color difference represented in the histogram.


#### 2. Explain how you settled on your final choice of HOG parameters.

The final feature vectors are composed by: spatial features, histogram features, and HOG features. The used color space for all features was `YCrCb`. After an exhausting hyperparamenter tunning process we decided that the best parameter combination for our problem was: 16 HOG pixels per cell, 2 cells per block, using all HOG channels, spatial binning of (32,32), 32 histograms bins and a histogram range of (0,256). All these variables are defined in the section 7 - Variables and data preparation of the IPython notebook (VehicleDetectionAndTracking_Final.ypnb).


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The implementation of the training phase is included in the Section 7 of the IPython notebook. We first loaded all the images and extracted the feature vectors of each one. We have 1125 non car images and 1196 car images, so we can say that we are facing a quite balanced dataset. The basic process in training a SVM model are: feature extraction, normalization and data preparation, training phase, testing phase and save the model. 

The feature extraction process is implemented in the Section 2 of the IPython notebook in the follwing order: HOG feature extraction, color histogram feature extraction, binned color features. The `extract_features` method extracts features from a set of input images meanwhile the `single_img_features`extracts the features from a single image. Both functions combined the three feature extractors we implemented. 

For the normalization we used a `StandardScaler`. The training process is located in an individual cell in order to avoid repeating the process. Also, the models are saved in the `data` folder.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding window implementation is located in the Section 3 of the IPython notebook called VehicleDetectionAndTracking_Final.ypnb. This is basically a function which generates in the image space several candidate windows defined by parameters such as window size, overlapping, and scale factor if we want multi-scale windows. In these windows we will proceed with the feature extraction process and then classification.

We used three different sliding windows with the following values:
* sliding window 1: x axis range (0, 1280), y axis range (400, 656), window size (48,48) and (0.5, 0.5) overalp.
* sliding window 2: x axis range (0, 1280), y axis range (400, 656), window size (64,64) and (0.6, 0.6) overalp.
* sliding window 3: x axis range (0, 1280), y axis range (500, 656), window size (96,96) and (0.7, 0.7) overalp.

As we can observed, we reduced the y axis range when the window size is quite big. This is because we are "looking closer into the image" so the cars should be bigger. At the same time the overlapping is increasing. The values was choosen after an exhaustive implementation (we tested with higher window sizes, different axis ranges and overlapping from 0.5 to 0.9). This is a result of bad values:

![alt_text][image4]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

These results are in three scales with different window sizes using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
Here are some example images:

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

The selection of a good window sizes and scales was critical in order to get a good results. Also, we improved this results using heatmaps avoiding in this way false positives. Also, the sliding window height is very important. We don't care about the sky and ground :).

Inspired in other works, we compute averaged heatmaps for n frames in order to prune the false positives. This worked very well in the final videos!

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In the previous figures we can see the results of this process.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult part of this project was choosing the sliding window parameters and also tunning the feature extraction stage. We improved our results by using heatmaps in order to avoid false positives. We also prune false positives in videos averaging those heatmaps during a number of frames. As feature work, we can integrate this project with the advanced line finding and also improve the sliding window implementation.
