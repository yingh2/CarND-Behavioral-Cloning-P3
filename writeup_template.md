# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_template.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The first thing is normalization to -1.0 and 1.0. Then I cropped out top 70 and bottom 25 pixels.
Then I added a convolution layer of filter 5 X 5 and depth 24. 
Then I added another convolution layer of filter 5 X 5 and depth 36.
Then another convolution layer of filter 5 X 5 and depth 48.
Then I dropped it with probability 0.2.
Then another convolution layer of filter 5 X 5 and depth 64.
Then another convolution layer of filter 5 X 5 and depth 64.
All convolution layers use RELU as well.
Then I added 4 dense layers of 100, 50, 10 and 1 with drop rate as 0.3 between each 2 layers.
#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center camera, left camera and right camera. I ran the simulator multiple times to get more data. Each run of the data result is saved in data?.zip. I also used the flip image trick to double the amount of data to be processed.

Because there is too much training data to load into memory once, I used a generator, this way, I can virtually have infinite amount of data without blowing up the memory limit.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a multi-layer CNN and Fully connected network.

My first step was to use a convolution neural network model similar to the network designed by Nvidia auto driving team.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has 3 drop out layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, I tried very lonng time and they all have some kind of error. I am running out of GPU time, so I can only submit the video with a few problems now.

#### 2. Final Model Architecture

The final model architecture (model.py lines) consisted of multiple convolution layers and Dense layers.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


#### 3. Creation of the Training Set & Training Process

After the collection process, I had 35900 X 3 (3 camera) number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rte wasn't necessary.
