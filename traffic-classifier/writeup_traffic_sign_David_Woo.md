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
[image7]: ./images/german_signs/classes_histogram.png "classes_histogram"
[image8]: ./images/german_signs/data_processing.png "data_processing"
[image9]: ./images/german_signs/confusion_matrix.png "confusion_matrix"
[image10]: ./images/german_signs/new_traffic_signs.png "new traffic signs"
[image11]: ./images/german_signs/new_images_and_class_probabilities.png "new traffic signs and predicted classes"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/david-woo-instacart/self-driving-cars/blob/master/traffic-classifier/Traffic_Sign_Classifier-v3_6-submission.ipynb)

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

![alt text][image7]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For image data, I did a few preprocssing steps
a) Used limited adaptive histogram equalization. This normalized the intensity or constrast across the image
b) Converted to greyscalce
c) Augmented image : rotate, scale, translate
![alt text][image8]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 greyscale image   					|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x250 	|
| Maxpooling         	| pool size (2,2)                           	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x500 	|
| Maxpooling         	| pool size (2,2)                           	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x750 	|
| Maxpooling         	| pool size (2,2)                           	|
| Fully connected		| Relu activation, weights = 300        		|
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
* training set accuracy of 0.997
* validation set accuracy of 0.9821
* test set accuracy of 0.970

Below is also the confusion matrix to further diagnose which signs the model does well with
![alt text][image9]

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



###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10]

As you can see the images, some of the images have objects in the background. This may make it harder for the model to classify as it won't be able to differentiate between foreground and background objects.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%, compared to the test set of 96% This seems low compared to the test set from the model. A possible explanation is the way I crop and resize the images or possibly i should distort the images that were used to train the model so that it can generalize more. Also, i noticed that the new traffic signs i introduced also have objects in the background. The model may not be able to differentiate the foreground (sign) and background

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below is a visualization of the top 5 softmax probabilities. The model seems to be biased towards no passing signs. This could be because this is a common occuring sign and the model may be defaulting to classifying as no passing.

![alt text][image11]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
