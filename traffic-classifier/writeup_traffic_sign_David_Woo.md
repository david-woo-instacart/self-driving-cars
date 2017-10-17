#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/german_signs/exploratory.jpg "Visualization"
[image2]: ./images/german_signs/blue_down_sign.jpg "Traffic Sign 1"
[image3]: ./images/german_signs/construction.jpg "Traffic Sign 2"
[image4]: ./images/german_signs/german_stop_sign.jpg "Traffic Sign 3"
[image5]: ./images/german_signs/road_slippery.jpg "Traffic Sign 4"
[image6]: ./images/german_signs/traffic_sign.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/david-woo-instacart/self-driving-cars/blob/master/traffic-classifier/Traffic_Sign_Classifier-v2.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart that shows the number of types of signs that are present in the training set. x-axis corresponds to the sign id and y-axis number of samples.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For image data, I normalized the data by perfoming (pixel - 128)/ 128.

I tried converting from RGB to greyscale but did not help the model, accuracy decreased

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x48 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x96 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x128 	|
| Maxpooling         	| pool size (2,2)                           	|
| Dropout           	| 0.5                                       	|
| Flatten				|												|
| Fully connected		| Relu activation        						|
| Dropout           	| 0.5                                       	|
| Fully connected		| Relu activation        						|
| Dropout           	| 0.5                                       	|
| Fully connected		| Softmax activation       						|
|						|												|
|						|												|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Type of optimizer = sgd. My choices were constant (SGD) vs adaptive step optimizers ( Adam). I choose a constant step optimizer or SGD. Since my network was a shallow net, i was think a constant step optimizer might be a better trade off between accuracy and performance.

Batch size = 32.

Epochs = 10

Hyper parameters : 1) learning rate = 0.01. Choose a small learning rate to increase probability of finding maximum or minimum point. Too large of a learning rate and the optimizer may not converge.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.968
* validation set accuracy of 0.9238
* test set accuracy of 0.92

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
My first architecture was:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Flatten				|												|
| Fully connected		| Relu activation        						|
| Fully connected		| Sigmoid activation       						|
|						|												|
|						|												|

* What were some problems with the initial architecture? I initially did not have a convolutional layer to begin with. I wanted to see how far I could get with a shallow network before adding complexity.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I added a convolutional network. My intuition is that with images this will keep the spatial stucture. ( minimize loss in infomation).

Also for the output layer instead of sigmoid I replaced it with softmax. sigmoid seems to be better for multi-label problems since the probability of each classes will be computed independently (i.e an increase in probability in one class does not decrease the probabillity in another class). However, since we are doing a multi-class classification a softmax seemed more appropiate. Since a increase in probability in one class would decrease probability in another class.

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4]
![alt text][image5] ![alt text][image6]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Turn right ahead      | Speed Limit   									|
| roadwork     			| Roadwork 										|
| Stop					| Slippery Road											|
| Slippery road	      	| Beware of Snow and Ice					 				|
| Traffic signals		| Priority Road      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This seems low compared to the test set from the model. A possible explanation is the way I crop and resize the images or possibly i should distort the images that were used to train the model so that it can generalize more.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 62th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.98         		| Priority Road      					    		|
| 0.00625166  	    | No Entry              							|
| 0.00269564  	    | Turn left ahead       							|
| 0.00106316  	    | Go straight or right   							|
| 0.00090599  	    | Roundabout mandatory							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
