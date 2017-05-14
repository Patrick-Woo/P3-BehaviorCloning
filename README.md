#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image0]: ./examples/steering_distribution.png "steering distribution"
[image1]: ./examples/nvidia_model.png "Model Visualization"
[image2]: ./examples/aug1.png "including left and right cameras"
[image3]: ./examples/aug2.png "brightness"
[image4]: ./examples/aug3.png "translation"
[image5]: ./examples/aug4.png "flip"
[image6]: ./examples/aug5.png "crop and resize"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_best-nvidia.json containing a trained convolution neural network structure
* model_best-nvidia.h5 containing network's weights
* README.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_best-nvidia.json
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5*5 filter sizes and depths between 24 and 64. 

The model includes RELU layers to introduce nonlinearity , and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually .

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see section below.

1. I use the keyboard to drive the car in the stimulator as I really don't have extra money to buy a ps4 controllor.

2.I drive the car with 5 round for counter-clockwise, 5 round for clockwise and 1 round for recovering from the edge to center. So, I collect 10 rounds data. And you can see the steer data distribution really balanced.

![alt text][image0]



###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the nvidia end2end model.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the offical nvidia end2end model

Here is a visualization of the architecture.

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 5 laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to get to the center when the vehicle left the center too much.

for more detailed explaination, please read the above section.

I use 4 methods to augment the data set,

Data Augmentation 1: Including Left and Right camera. 

I use the shift steering angle value of 0.25 for the left and right camera.

![alt text][image2]

Data Augmentation 2: Brightness shifting

![alt text][image3]

Data Augmentation 3: translation

![alt text][image4]

Data Augmentation 4: Flipping images

![alt text][image5]

After the collection process, I had 8400 number of data points. I then preprocessed this data by crop the image and then resize to 64*64.

![alt text][image6]


I finally randomly shuffled the data set by generator. also I use the generator for generating the validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 60 as evidenced by mse is the least. I used an adam optimizer so that manually training the learning rate wasn't necessary.
