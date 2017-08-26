# **Traffic Sign Recognition** 
### Khanh Nguyen (nguyen.h.khanh@gmail.com)
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is __34799__
* The size of the validation set is __4410__
* The size of test set is __12630__
* The shape of a traffic sign image is __(32,32,3)__
* The number of unique classes/labels in the data set is __43__

#### 2. Include an exploratory visualization of the dataset.

A bar chart showing the count for each class in the training set is included in the notebook.

5 most common signs in the training, more than 1750 examples, are: 
- Speed limit (50km/h) (class id #2)
- Speed limit (30km/h)  (class id #1)
- Yield  (class id #13) 
- Priority road  (class id #12)
- Keep right  (class id #38)

5 least common signs in the training data, with less than 250 examples, are: 
- Speed limit (20km/h) (class id #0) 
- Dangerous curve to the left (class id #19) 
- Pedestrians (class id #27) 
- Go straight or left (class id #37) 
- End of no passing (class id #41)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My preprocessing pipleline is as follow:

1) Oversampling using SMOTE. After this, the number of training examples in each class are equal. 
2) Convert image to grayscale.
3) Normalize pixal using formula (pixel - 128)/128

Step 2) and 3) are implemented inside Lenet(x). 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is LeNet from the lecture with an additioanl Dropout at the last convolutional layer.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten				| output 400									|
| Fully connected		| output 120        							|
| RELU					|												|
| Fully connected		| output 84         							|
| RELU					|												|
| Dropout       	    | 0.5       									|
| Fully connected		| ouput 43     									|
| Softmax				|           									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I tweak the original hyperparameters a bit. I increase the number of epoch to 20, learning rate to 0.002, and add a dropout prob 0.5

I experimented with higher epoch numbers, thinking it will improve my result but I couldn't achieve a statistical signficant improvement so I keep it at 20.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.7%
* validation set accuracy of 95.3% 
* test set accuracy of 93.8%

I chose LeNet architecture for the problem because the training examples are similar in nature. The number of classes is in the same order of magnitude. I belive the model works well because it achieves accuracy > 93% only after 20 epochs. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I picked 20 images from https://github.com/navoshta/traffic-signs/tree/master/traffic-signs-data/custom

Here are some images from the set

![alt text](./new_images/example_00001.png)
![alt text](./new_images/example_00002.png)
![alt text](./new_images/example_00003.png)
![alt text](./new_images/example_00004.png)
![alt text](./new_images/example_00005.png)

Image #1 might be difficult to classify because the the sign (inside the triangle) is small, making it hard to detect distinct feature even with human eyes.

![alt text](./new_images/example_00011.png)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

My model predicts with 90% accuracy. 2 out of 18 are incorrect. This is comparable with the validation and testing accuracy, taking into account the fact the accuracy can swing significantly in a small sample. Another incorrect classification could bring the accuracy to 83%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The probabilities are shown in the notebook output. 

In #11, the model mistakes a children crossing for a bicycle crossing. When we exam the probability ranking, the order is [Bicycles crossing, Children crossing, Slippery road,Beware of ice/snow,Road narrows on the right]. Not too far from the truth.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

