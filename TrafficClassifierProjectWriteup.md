#**Traffic Sign Recognition** 

---

[//]: # (Image References)

[visualization]: ./report/visualization.png "Visualization"
[grayscale]: ./report/grayscale.png "Grayscaling"
[augmentation]: ./report/augmentation.png "Random Noise"
[new_raw_processed]: ./report/new_raw_processed.png "New images raw & processed"
[incorrect]: ./report/incorrect.png "Incorrectly Classified"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is *34799*
* The size of test set is *12630*
* The shape of a traffic sign image is *(32, 32, 3)*
* The number of unique classes/labels in the data set is *43*

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. The visualization shows a sample image for each traffic sign ordered in descending order by the frequency count in the training dataset.

![visualization][visualization]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, the images were converted to grayscale to reduce processing power as recommended in the Sermanet/LeCunn paper and it does not look like any signal is missed by reducing to grayscale.
The nest step was to normalize the bring image data to the same range and enable the network to generalize.

Here is an example of a traffic sign image before and after grayscaling.

![grayscale][grayscale]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for loading the data is contained in the first code cell of the IPython notebook.  
The training data is already split into a validation data and is provided as part of the initialization step.

My final training set had 60567 number of images. My validation set and test set had 4410 and 12630 number of images.

The seventh and eigth code cells of the IPython notebook contain the code for augmenting the data set. I decided to generate additional data to simply have more training data for the model to train on. The initial model with just normalization and gray scaling and with a standard LeNet architecture reported a validation accuracy of 92%. Data augmentation is also a recommended technique and has been used by many of the state of the art models. To add more data to the the data set, I used the following techniques:
    * rotate all images in a given batch between -10 and 10 degrees
    * random translations between -10 and 10 pixels in all directions.
    * random zooms between 1 and 1.3.
    * random shearing between -25 and 25 degrees.

Augmentation was done only for 26 traffic signs whose count in the data was less than 800 (the mean image count in the data) and the jittering was done for each image in those traffic signs effectively doubling the occurrence count of these traffic signs. 
Here is a visualization showing an original image and an augmented image for each of the 26 traffic signs after sugmentation in the training data:

![augmentation][augmentation]


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GRAYSCALE image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64    				|
| Flatten        		| input 5 * 5 * 64 layer output 1600            |
| Fully connected		| output flat layer with 1024                   |
| RELU					| probability 0.8								|
| DROPOUT				|												|
| Fully connected		| output flat layer with 1024                   |
| RELU					|												|
| DROPOUT				| probability 0.8								|
| SOFTMAX				|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the fourteenth cell of the ipython notebook. 

To train the model, I used AdamOptimizer since it converges faster with less tuning because it uses momentum with decay and with the following hyperparameter settings:
* batch size 100
* number of epochs 60
* learning rate 0.0009

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the fourteenth, fiftenth, sixteenth cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 96.0 %
* test set accuracy of 94.1 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

    * The very first architecture tried was a standard LeNet model with pre-procesing of the images.

* What were some problems with the initial architecture?

    * The validation accuracy of the model was decent at around 92 %. The validation accuracy seemed low and with many well-known still to be explored.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

    * By augmenting the data as explained above
    * Increasing the channels for the convolution kernels and   increasing the width of the fully connected layers.
    * Adding a dropout layer after each of the fully connected layers.

* Which parameters were tuned? How were they adjusted and why?

    * The parameters tuned were batch size, learning rate and the number of epochs. The batch size and the learning rate parameters were brought down a little and the number of epochs increased to get a better validation accuracy. The validation accuracy was plotted by the number of epochs to view the progress. The visualization showed a flattening curve at which the optimization was stopped. The curve is still has a lot of jitter and it shows that the size of the validation set is smaller and could have been improved by adding it to the validation set already provided. But I decided to not tinker with it has its already been provided.
 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

    * Without adding a dropout layer the model clearly was over-fitting and the validation accuracy tended to go down after 25-30 epochs. To avoid this the dropout layers were added after the 2 fully-connected layers which helped in achieving somewhat of a stable state.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I chose to download 16 images, most from wikipedia and some from generally from search results. The wikipedia images were the same size but the ones downloaded were quite bigger size. Out of 12 wikipedia images 10 were classified correctly. There were 2 images that were not classified correctly and all the 4 bigger size images were classified incorrectly.

Here are the German traffic signs that I use from the web:

The visualization shows the images used for predictions
![New raw images and processed][new_raw_processed]

The images from the smaller one that could not be classified were the road work ones. Maybe because there were not enough road work images in the dataset to acocunt for the flipped ones. So, augmenting the data with flips could have resolved the issue.
The other bigger sized images when down scaled to 32 x 32 loose lot of information and that is why were not correctly classified.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

Here are the results of the prediction:

| Prediction			                   |     Image	        			        | Top probability |
|:----------------------------------------:|:--------------------------------------:|:---------------:|
| Speed limit (60km/h)      		       | Speed limit (60km/h)              		| 1 |
| No Passing    			               | Pedestrians						    | 1 |
| Dangerous curve towards the right		   | Road Work								| 1 |
| Right-of-way at the next intersection	   | Right-of-way at the next intersection	| 1 |
| Keep right			                   | Keep right          					| 1 |
| Speed limit (20km/h)                     | Road Work                              | 1 |
| Speed limit (30km/h)                     | Speed limit (30km/h)                   | 1 |
| Slippery Road                            | Speed limit (60km/h)                   | 1 |
| Slippery road                            | Children crossing                      |.984|
| Yield                                    | Yield                                  |.989|
| General caution                          | General caution                        | 1 |
| Stop                                     | Stop                                   | 1 |
| Stop                                     | Stop                                   | 1 |
| Road work                                | Speed limit (30km/h)                   |.892|
| Priority road                            | Priority road                          | 1 |
| Turn left ahead                          | Turn left ahead                        | 1 |

The model was able to correctly guess 10 of the 16 traffic signs, which gives an accuracy of 62.5%. If we only consider the smaller images then it correctly guessed 10 out of 12 signs which gives an accuracy of 83.3 % and is still quite less than the accuracy on the test set of 94.1 %.
The thing to notice about the above probabilities is that the model is very certain about its predictions where it gives 100% to the 1st prediction on the list.
There are couple of images where the correct option is in the top 5 though the probabilities assigned to them are very less:
* Children crossing - The correct option is the 2nd in the list.
* Speed limit (60km/h) - The correct option is 3rd in the list.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th and 20th cell of the Ipython notebook.
Here I concentrated only on the images predictions of which are wrong. Looking at the top 5 predictions of each image and the correct label the predictions were actually horribly wrong with none of the predictions in the top 5 indicating the correct label.

The visualization shows the images with incorrect predictions and their top 5 probabilities.
![incorrect][incorrect]
