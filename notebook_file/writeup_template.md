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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ../traffic-signs-data/1.png "Traffic Sign 1"
[image5]: ../traffic-signs-data/2.png "Traffic Sign 2"
[image6]: ../traffic-signs-data/3.png "Traffic Sign 3"
[image7]: ../traffic-signs-data/4.png "Traffic Sign 4"
[image8]: ../traffic-signs-data/5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/airbird0421/traffic-sign-detection/edit/master/notebook_file/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Basically, I used numpy to get summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

First, I printed out how many samples for each class_id in the training set, as following (in class_id increasing order):
[180, 1980, 2010, 1260, 1770, 1650, 360, 1290, 1260, 1320, 1800, 1170, 1890, 1920, 690, 540, 360, 990, 1080, 180, 300, 270, 330, 450, 240, 1350, 540, 210, 480, 240, 390, 690, 210, 599, 360, 1080, 330, 180, 1860, 270, 300, 210, 210]

Second, I showed 10 sample pictures, one for each class, in the ipynb file and the report html file. (ipynb gives warning if I show all 43 classes)

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I normalized the image data with the suggested methord, i.e. (pixel - 128) / 128, so that the input has 0 mean and small variance.

Second, I decided to convert the images to grayscale. The reason is that, traffic sign has nothing to do with color and changing to one-channel image can reduce the intial number of feature maps and seems improving the validation accuracy with my training model. But I didn't really convert the images to grayscale, I only took the mean of the 3 channels as the 1-channel value, which should be similar to grayscale.  

From above statistics, I know the numbers of training samples of different classes are different, varing from 180 to 2010. But I didn't add additional data to the training set. Instead, based on some search results from internet, considering the training samples are relatively huge, I used the so-called under-sampling method. I.e. for each epoch, I don't take all the training samples, instead, I take the samples in a way such that for each class, it's no more than 1500. I also tried different numbers here, like 500, 1000, etc. But I don't see quite obvious improvement with this methord. I still used it since it seems giving a little better result.

I also tried other preprocessing methords, like rotating some of the images, but didn't get ideal results. So I didn't include it here. I don't know if this is because I did it the wrong way or something else. 

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Fully connected		| input 800, output 120        									|
| RELU					|												|
| dropout					|			keep_prob = 0.5									|
| Fully connected		| input 120, output 84        									|
| RELU					|												|
| dropout					|			keep_prob = 0.5									|
| Fully connected		| input 84, output 43        									|
| Softmax				| etc.        									|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam Optimizer, and batch size 128, 20 epochs, and 0.5 dropout rate, 0.001 learning rate.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.4% (I deleted the code to calculate the accuracy for the training set since I basically followed the code structure of LeNet example. The calculation mothod is basically same as test set accuracy)
* validation set accuracy of 96.9
* test set accuracy of 94.9

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The LeNet architecture. The reason is that it's a great architecture for classifying images, which is just the purpose for this project.
* What were some problems with the initial architecture?
Some small ajdustments are needed, like the input/output size for each layer, and the main problem is that the accuracy is only about 89%
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Basically, since the initial accuracy is low, I think it's because of underfitting. I think the reason should be that the traffic data sets have more classes, and the traffic signs are more complicated picutres than numbers, which contain more features. So the first thing I did is to add more parameters, i.e. increase the feature map numbers of the convolution layers, to 12 and 32, doubling the default numbers of LeNet. The accuracy increased to about 93% after this. Since accuracy got improved, I tried to add dropout to prevent overfitting. With a dropout rate of 0.5, the accuracy becomes above 95%

* Which parameters were tuned? How were they adjusted and why?
After above mentioned two measures, adding more feature maps, and adding dropout layers, I also tried to tune other parameters, like learning rate, batch size, epochs. I increased epochs from 10 to 20, and the accuracy increased to above 96%. But with different learning rate, batch size, I either see worse results or no obvious improvements.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
This is basically explained above in the architecture adjustment topic. The most important two measures are add more feature maps to the convolution layers and adding dropout layers. Also, increase epoch number seems helpful.

If a well known architecture was chosen:
* What architecture was chosen?
The LeNet architecture.
* Why did you believe it would be relevant to the traffic sign application?
First, it's a well known architecture to classify images, which is just the task of this project. Second, even with its default architecture, the accuracy is already very high, 89%, which indicates that with some adjustments, it should be able to fullfil the requirements of this project.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The accuracy of all the data sets is about 95% or above, which means the model is working well, maybe not perfectly. There's still overfitting problem since the accuracy is lower on validation and test sets.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![go straight or left][image4] ![roundabout mandatory][image5] ![speed limit (60 km/h)][image6] 
![slippery road][image7] ![traffic signals][image8]

I thought the 4th image (slippery road) may be difficult to classify since it looks that the picture contains more "features", but it turned out that the model did wrong with the 5th.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| go straight or left      		| go straight or left   									| 
| roundabout mandatory     			| roundabout mandatory 										|
| speed limit (60 km/h)					| speed limit (60 km/h)										|
| slippery road	      		| slippery road					 				|
| traffic signals			| general caution      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The accuracy on the test set of 94.9%. I think maybe providing more new images can get a hiher accuracy. I checked why the 5th (traffic signals) is predicted wrong. One reason should be that the traffic signal class only have about 500 samples, which is relatively lower that many other classes. The other reason is that it does look similar to the general caution sign.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 55th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a "go straight or left sign" (probability of 1.0!!), and the image does contain a "go straight or left" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| go straight or left   									| 
| 1.026e-8     				| trun right ahead 										|
| 1.36e-9					| keep left											|
| 6.9e-10	      			| General caution					 				|
| 4.72e-11			    | Yield      							|


For the second image the model is pretty sure that this is a "roundabout mandatory" (probability of 0.999!!), and the image does contain a "roundabout mandatory" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| roundabout manadatory   									| 
| 2.24e-4     				| priority road										|
| 3.09e-6					| Beware of ice/snow											|
| 9.76e-7	      			| right-of-way at the next intersection				 				|
| 2.16e-7			    | speed limit (100km/h)    							|


For the third image the model is pretty sure that this is a "speed limit (60km/h)" (probability of 0.999!!), and the image does contain a "speed limit (60km/h)" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| speed limit (60km/h)   									| 
| 9.04e-5     				| speed limit (80km/h)									|
| 1.93e-7					| End of speed limit (80km/h)						|
| 1.06e-7	      			| go straight or right				 				|
| 8.85e-8			    | dangerous curve to the right     							|

For the fourth image the model is pretty sure that this is a "slippery road" (probability of 0.999!!), and the image does contain a "slippery road" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| slippery road   									| 
| 5.26e-6     				| Beware of ice/snow									|
| 7.69e-9					| dangerous curve to the right											|
| 2.83e-10	      			| children crossing					 				|
| 2.19e-11			    | dangerous curve to the left     							|

For the fifth image the model thinks this is a "General caution", but the image is actually a "traffic signal" sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999        			| general caution  									| 
| 8.55e-5     				| traffic signal									|
| 7.77e-12					| Pedestrians											|
| 4.32e-15	      			| 	Right-of-way at the next intersection				 				|
| 4.30e-19			    | road narrows on the right    							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I visuallized all the 5 new images. Basically, it seems that the convolution layer 1 can already recognize the exact traffic signs, including the outside circle or triangle, and the inside actual signs. convolution layer 2 maybe too small (only 5 x 5), so visulization doesn't provide much information.


