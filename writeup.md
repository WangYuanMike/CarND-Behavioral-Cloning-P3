#**Behavioral Cloning** 
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/WangYuanMike/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [model.ipynb](https://github.com/WangYuanMike/CarND-Behavioral-Cloning-P3/blob/master/model.ipynb) containing the training process and output
* [drive.py](https://github.com/WangYuanMike/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode (I did not change it for the NVIDIA model)
* [model.h5](https://github.com/WangYuanMike/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network 
* [run1.mp4](https://github.com/WangYuanMike/CarND-Behavioral-Cloning-P3/blob/master/run1.mp4) recording video of my vehicle driving autonomously for more than one lap around the track
* [writeup_report.md](https://github.com/WangYuanMike/CarND-Behavioral-Cloning-P3/blob/master/writeup.md) summarizing the results (You are reading it!) 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the model from NVIDIA paper. Here are the details of model layers.

| Layer         		       |     Description	        					                 | 
|:------------------------:|:---------------------------------------------:| 
| Input         		       | 160x320x3 BGR image   				               | 
| Lambda         		       | Normalization                               | 
| Cropping         		     | Cropping2D(cropping=((70, 25), (0, 0)))      | 
| Convolution 5x5     	   | 2x2 stride, valid padding, filters 24, ReLU  	 |
| Convolution 5x5     	   | 2x2 stride, valid padding, filters 36, ReLU  	 |
| Convolution 5x5     	   | 2x2 stride, valid padding, filters 48, ReLU  	 |
| Convolution 3x3     	   | 1x1 stride, valid padding, filters 64, ReLU  	 |
| Convolution 3x3     	   | 1x1 stride, valid padding, filters 64, ReLU  	 |
| Flatten                  |                                    |
| Fully Connected          | outputs 100                |
| Fully Connected          | outputs 50                |
| Fully Connected          | outputs 10                |
| Fully Connected          | outputs 1 (the predicted steering angle)      |

#### 2. Attempts to reduce overfitting in the model

I did not use any specific step (e.g. regularization, dropout) to reduce overfitting, because the car can drive smoothly with the model I mentioned above. But I used data augmentation through generator. Please check details in chapter below.

#### 3. Model parameter tuning

| Hyperparameter         		| Final Value      					                        | 
|:------------------------:|:---------------------------------------------:| 
| Initializer              | xavier_initializer                            |
| Optimizer     	          | Adam                                          |
| Learning rate       	    | 1e-3                                      	   |
| Learning rate decay rate     | 1e-2                                   |
| batch size                | 64                                  |
| Epochs                   | 10                                           |

#### 4. Appropriate training data

I just used the sample data provided by udacity. First split the data set into training set(80%) and validation set(20%). I took all center, left, and right images with corrected steering angles. I also flipped all images in training set for data augmentation. All of the data loading and augmentation are done in generator, and of course I shuffled the data in the generator.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I do not have so much to say in this part, as I just followed the NVIDIA paper to build the model and the udacity lecture to load and augment the data with generator. The hyperparamenters are chosen by experience, and luckily the model work well on the simulator after 10 epochs training. As the model training went pretty well, I did not add any additional steps to tune the model further, e.g. record specific training data, or adding dropout layer.
I have tried most of the model designing and training approach from the [last project](https://github.com/WangYuanMike/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md), so I decided to put more effort on transfer learning in this project.

#### 2. Transfer Learning
I took some time trying out transfer learning using bottleneck feature, but I did not get a successful simulation result on my laptop. [model.ipynb on ResNet branch](https://github.com/WangYuanMike/CarND-Behavioral-Cloning-P3/blob/ResNet/model.ipynb) is the main file for ResNet transfer learning. And here are my findings and questions.
* The model.h5 file of transfer learning is pretty big, probably because it includes the weights of ResNet. In my case it is about 94 MB, therefore it is not very convinient to use github and git to transfer it between GPU instance and my laptop, although it is possible. Finally I used scp to transfer it and exclude it from github repository. Do you have any suggestion on handling big model file?
* I am not 100 percent sure whether I used a right way or at least an elegant way to build the final model which consists the data preprocessing layers, resnet(without the top layers), and the shallow fully connected model that I trained based on the bottleneck features. I am not sure about the way of resizing the input image from (160, 320) to (224, 224), which is the input size of ResNet. I used a tensorflow method to do it, and correspondingly I have to import tensorflow in [drive.py](https://github.com/WangYuanMike/CarND-Behavioral-Cloning-P3/blob/ResNet/drive.py) to make the model work in simulation. Would you please share some idea on building the complete model of transfer learning? And do you have any suggestion on the way that I generates the bottleneck features. 
* Well, when I really tested the ResNet transfer learning model on my windows laptop, I found that the main problem could be the computing resource limitation of my laptop. From the console, I can see that it generates only about two throttle/angle pairs per second, which caused the car always drove with some outdated instructions (instructions based on previous images). In the NVIDIA model, the model is much simpler (about 4 MB) and faster, and it generates scores of throttle/angle paris per second, which makes the cars drives very smoothly on a speed of 9 mph. So actually on my laptop, I could not verify whether the transfer training model works fine . Do you have any idea on dealing with this low speed simulator issue?
* I tried to adjust some parameters in the drive.py, e.g. set_speed = 4, Kp = 0.01. And to some extend, it helps to mitigate the issue, namely the car can drive longer on the track, but still not as smoothly as the NVIDIA model. Do you have any suggestions on adjusting these parameters? Plus, would it be helpful to also generate throttle from the model instead of generating it in the drive.py? 
