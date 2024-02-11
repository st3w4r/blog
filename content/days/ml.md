---
title: "Learning Machine Learning"
date: 2022-08-09T00:00:00Z
draft: false
type: "day_to_day"
number_of_days: 34
updated_on: false
---

Machine Learning and Deep Learning are trendy topics and it is a promising area. As a Software Engineer, it's identified as a Data Scientist subject but it sounded so cool that I decided to dig it. Here are my day-to-day notes on this long journey to master the field.

---

[March 2020 - 13 days]({{< ref "/days/ml#2020-03-17">}})\
[April 2020 - 21 days]({{< ref "/days/ml#2020-04-01">}})\
[next]({{< ref "/days/ml#2020-04-30">}})

---


## [2020-03-17]({{< ref "/days/ml#2020-03-17" >}}) {#2020-03-17}


Hello world for CNNs, make a simple network that predicts the MNIST digits

[https://keras.io/examples/mnist_cnn/](https://keras.io/examples/vision/mnist_convnet/)

- check out how the model is built after you compile it with: model.summary()
- try changing a few parameters in the model and check out how it looks now
- when you train the model, change some of the factors like: batch_size, epochs and learning rate
- print out how the data looks for the different pre-processing steps
- Start lesson 1 [fast.ai](http://fast.ai)


## [2020-03-18]({{< ref "/days/ml#2020-03-18" >}}) {#2020-03-18}

Tasks: 

- [x]  Continue [lesson 1 fast.ai](https://course19.fast.ai/videos/?lesson=1)
- [ ]  Train model on MNIST dataset

I have seen how to download a dataset, how to create a DataLoaders and how to train a pre-trained model.

In the learner, I see different types of layers: Conv2d, BatchNorm2d, ReLU, MaxPool2d, etc.

Questions:

- What does freeze and unfreeze?
- What's a kernel?
- What's a sequential model?
- What does each layer do?


## [2020-03-19]({{< ref "/days/ml#2020-03-19" >}}) {#2020-03-19}

Try to tweak a model.  
Read the documentation, and discover the simple model creation.  
The list of definitions: [https://deepai.org/definitions](https://deepai.org/definitions)

Questions:

- What's a kernel?  
    Use to map a non-linear problem to a linear methods
- What's the stride?  
    Modify the amount of movement over the image.
    If stride set to 1 the filter will move one pixel or unit at a time.  
    [https://deepai.org/machine-learning-glossary-and-terms/stride](https://deepai.org/machine-learning-glossary-and-terms/stride)
- What is an epoch? Cycles through all the data
- What are callbacks in the training loop for?
    - Used to tweak the training loop at different stages: after epoch, after loss, before validate, etc
- What's one cycle?
    - [What is 1cycle?](https://fastai1.fast.ai/callbacks.one_cycle.html#The-1cycle-policy)
    - [A disciplined approach to neural network hyper-parameters: Part 1 -- learning rate, batch size, momentum, and weight decay](https://arxiv.org/pdf/1803.09820.pdf)
- What's the learning rate per iterations


## [2020-03-20]({{< ref "/days/ml#2020-03-20" >}}) {#2020-03-20}

Read more documentation about the [callbacks](https://docs.fast.ai/callback.core.html), the callback makes it easy to add new functionalities to the training loop.

Read about the learning rate.

[Setting the learning rate of your neural network](https://www.jeremyjordan.me/nn-learning-rate/)

- A small learning rate will require many updates before reaching the minimum point
- Optimal learning rate swiftly reaches the minimum point
- Too large learning rate will cause drastic updates which lead to divergent behaviors

Test all three phases to discover the optimal learning rate range.

[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) (Paper)

[Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) (Paper)

Questions:
- What's a hyper-parameter?


## [2020-03-21]({{< ref "/days/ml#2020-03-21" >}}) {#2020-03-21}

I have watched videos of [Three Blue One Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) about Neural Networks.

A great method to train a model is the one-cycle method.

[A disciplined approach to neural network hyper-parameters](https://arxiv.org/abs/1803.09820) ([Paper](https://arxiv.org/pdf/1803.09820.pdf))


ReLU (Rectified Linear Unit) is an activation function.

Question:
- How does ReLU work behind the scene?
    [Rectifier (neural networks)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))


Tasks:
- [x]  Train a model with a custom dataset


## [2020-03-22]({{< ref "/days/ml#2020-03-22" >}}) {#2020-03-22}

I have trained a model with my own dataset. I have to clean up the top losses data, Fastai provides a class to do that. Load the new data and retrain the model.

I have to export the model, it creates a pickle file `.pkl`

This file contains the model, the weights, metadata like the classes or the transforms

I have built a small program that takes this file and loads it with Fastai and returns the prediction on custom images.

Now I have a good idea of how to export a model and run it into production.

"Inference" is the name of the prediction process.

Learning rate too high -> The validation loss will be high

Learning rate too low the error rate will decrease very slowly between epochs. An indication is the training loss will be higher than the validation loss

To recognize if a model is starting over fitting is only if the error rate starts to be worth than the previous epochs. Not about training loss lower than validation loss.

Tensor with 3 dimensions for a colored image.

Visualize [Matrix Multiplication](http://matrixmultiplication.xyz)

Acronyms:
- SGD: Stochastic Gradient Descent
- MSE: Mean Squared Error

Learning linear regression and the derivative is useful to understand how loss calculation works.

To calculate the gradient descent in practice, we use mini-batches instead of calculating the whole batch.

Vocabulary:
- Learning rate\
    Multiple the gradian by.
- Epoch\
    One complete run all over the data point (image, ...)\
    E.g. if 1K images, mini-batch is 100 it will take 10 iterations to see every image once, for 1 epoch.\
    Too many times looking at the same image can lead to overfitting.
- Minibatch\
    Random data points to update the weights.
- SGD\
    Stochastic Gradient Decent using mini-batches
- Model / Architecture\
    Functions
- Parameters\
    Coefficient → Weights
- Loss function\
    How far away or close you are from the correct answer.


To sharpen your math understanding
- [There's no such thing as "not a math person"](https://www.youtube.com/watch?v=q6DGVGJ1WP4)
- [Khan Academy](https://www.khanacademy.org/)


[How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/)

Questions:
- What's gradient?
- What's a tensor?
- How to embed the model into an application?

Tasks:
- [x]  Export the model
- [x]  Embed the model into a python script to process one image


## [2020-03-23]({{< ref "/days/ml#2020-03-23" >}}) {#2020-03-23}

Today learning about computer vision and image processing.
- A black and white image is a bunch of numbers representing the grayscale.
- A colored image has 3 dimensions, named channels represented as a rank 3 tensor.
- 3D tensor (red, green, blue)
- A tensor is an array with a regular shape, where every row is the same length and every column is the same length. Mathematically, we always go with number of rows by the number of columns.

**Rank:** How many dimensions are there. Colored image has a rank of 3.

[Visualize Neural Networks](https://distill.pub/2020/grand-tour/)

---

[Computational Linear Algebra for Coders](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md)

Linear Function:
```
y = ax+b
```

```
x, y	are the coordinates of any point on the line
a	is the slope of the line
b	is the y-intercept (where the line crosses the y-axis)
```


## [2020-03-24]({{< ref "/days/ml#2020-03-24" >}}) {#2020-03-24}


[Fast.ai Lesson 2](https://course19.fast.ai/videos/?lesson=2) at [Linear Regression Problem](https://youtu.be/ccMHJeQU4Qw?t=5241)

Find the parameters to minimize the error. For a regression problem, the error function a.k.a **loss function** MSE is a common one.

Mean Squared Error → MSE, also RMSE → Root Mean Squared Error

MSE is the loss, the difference between predictions and the actual number.


## [2020-03-25]({{< ref "/days/ml#2020-03-25" >}}) {#2020-03-25}

Continue the calculation of the SGD.

Create mini batch to don't train on the whole dataset each time before updating the weights.


## [2020-03-26]({{< ref "/days/ml#2020-03-26" >}}) {#2020-03-26}

Starting [lesson 3](https://course19.fast.ai/videos/?lesson=3) about data blocks, multi-label, classification, and segmentation.

`thresh` is used in case we have multiple labels.

Python `partial` is used to create a function with a specific parameter.

Use small images to experiment more quicker.

Segmentation creates a labialization pixel per pixel.

- Difference between online and offline ML
    - Offline: Static dataset
    - Online: Continuously train when the data comes in. a.k.a incremental learning

## [2020-03-27]({{< ref "/days/ml#2020-03-27" >}}) {#2020-03-27}

Using progressive resizing to train the model.

If underfitting, train longer, train last part reduce learning rate, decrease regularization.

U-Net for segmentation training

Learning rate should be high at the begging and reduce after. Don't be locked in to find the smallest.

After each linear regression use an activation function.\
The sigmoid is not used anymore.\
Mostly used now:\
ReLU: Rectified Linear Unit → ReLU activation\
`max(x,0)`

[A visual proof that neural nets can compute any function](http://neuralnetworksanddeeplearning.com/chap4.html)\
[Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

Imagenet expect 3 channels, if we have only 2 channels we can create a new channel set as 0.

Tasks:
- Practice data block API
- Multi label classification
- Segmentation
- NLP classification
- Implement MSE
- Practice ReLU

## [2020-03-29]({{< ref "/days/ml#2020-03-29" >}}) {#2020-03-29}

Watched the videos of [3blue1brown - Essence of linear algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) about linear algebra, vector, linear transformation, matrix multiplication.

Now working on fastai and data blocks API.

What's the normalization imagenet_stats

Questions:

- What's BatchNorm2d
- What's dropout

## [2020-03-31]({{< ref "/days/ml#2020-03-31" >}}) {#2020-03-31}

Working with Fastai library, and practice the data block API.

I read some articles [Basics of Linear Algebra for Machine Learning](https://machinelearningmastery.com/linear_algebra_for_machine_learning/) super valuable, he advises the top-down approach and prevents mistakes of what beginners do. They learn math too early in the ML journey.

ResNet solve the vanishing gradient

Article: [Deep Convolutional Neural Networks](https://towardsdatascience.com/deep-convolutional-neural-networks-ccf96f830178)

CNN: Convolutional Neural Network

Question:
- What's a convolution?\
    Convolution operation consists on passing a kernel over an input volume.\
    Used to capture features.

## [2020-04-01]({{< ref "/days/ml#2020-04-01" >}}) {#2020-04-01}

Want to work with a segmentation dataset, importing this COCO datasets http://cocodataset.org/

Train the model.

Python division with two slashes to do integer division `10//3 = 3` 

Questions:
- What's a mask exactly?
- How to create a fallback in an image classification? If not detected go to the fallback case, should we implement that post classification based on the minimum score?
- What's the segmentation used for? Advantage?
- How to evaluate the memory footprint to guess the batch size?


## [2020-04-02]({{< ref "/days/ml#2020-04-02" >}}) {#2020-04-02}

I continue to train on CamVid.

Question:
- What unet is?


## [2020-04-05]({{< ref "/days/ml#2020-04-05" >}}) {#2020-04-05}

Working on segmentation, based on the camvid tiny dataset.

I'm exploring the mask and the function to display it. It call an external library and the convert mode is used to display different color based on the value in the image. The value does not represent an RGB value.

I check the min value and the max value in the tensor

```python
torch.min(mask.data)
torch.max(mask.data)
```

I expect it should match the number of codes I have in my dataset.

It will depend on the image, not on all the images we have all the things in the label.
I will check how many different value I have in my dataset instead min and max.

```python
len(np.unique(mask.data))
```

Now it make sense, I can discover all the things I have in the images based on the mask code.

I trained the model, still learning about the learning rate and the plot loss.

Tasks:
- Train a model with one class only
- Prepare a small dataset with multi classes
- Train my model on the COCO datasets.


Starting fast.ai lesson 4

NLP:
Legal Text classifier with univeral model fine tuning.

[Universal Language Model Fine-tuning for Text Classification (paper)](https://www.aclweb.org/anthology/P18-1031.pdf)

NLP use transfer learning, use pre trained Language model from wikipedia Wikitext.

Language model → Specific language model → Classifier\
Self supervised learning


Collaborative filtering:\
[Sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) storage\
Cold start problem, to solve use a metadata model

Terms:\
**Parameters/weights** → Number inside the matrices\
**Activations** → Result of the calculation (Result of matrix multiply or  activation function)\
**Layers** → Each step of the computation\
**Input** → Entry point\
**Output** → Result

Common to have a sigmoid at the last layer, to have an output between 2 values

Loss function compare the result with the last layer.


## [2020-04-06]({{< ref "/days/ml#2020-04-06" >}}) {#2020-04-06}

Training on the head poses dataset.

Data augmentation can help generate more data for the training set.

## [2020-04-08]({{< ref "/days/ml#2020-04-08" >}}) {#2020-04-08}

Working with tabular data today.

- Continuous: Represents values without limits, like age.
- Categorical: Limited to a small set of possibilities.

For categorical data, we will use embeddings.

A processor, similar to transformations in computer vision, will be used, but it's applied beforehand.

The validation set should be a contiguous group of items.


## [2020-04-10]({{< ref "/days/ml#2020-04-10" >}}) {#2020-04-10}

Regarding Collaborative Learning, what is the difference between Embedding NN and Embedding Dot Bias?

Will fastai keep the better parameters of the training when we change the learning rate?

How are columns selected in CollabDataBunch?

What is PCA?

Calculate the loss with RMSE (Root Mean Squared Error).

## [2020-04-11]({{< ref "/days/ml#2020-04-11" >}}) {#2020-04-11}


I'm watching a [course from MIT](https://www.youtube.com/watch?v=iaSUYvmCekI&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=3) about computer vision CNNs.

Layers:

- Conv
- Relu
- MaxPooling: Takes the maximum value in the window

Apply layers to match the depth of the neural network.

Convolution:

- Applying a window of weights
- Computing linear combinations
- Activating with a nonlinear function

Depth: h x w x d

Create a feature map.

Non-Linear Activation:

- Applying activation function: ReLU `max(0,z)`. Values lower than zero are set to 0; values greater than 0 remain unchanged.

Pooling:

- Takes the maximum value of the patch, shrinking the image
- Patch

CNN:

- Feature Learning
- Classification

Tensorflow provides a [playground](http://playground.tensorflow.org/) tool.

The playground is pretty nice; you can better understand the learning rate, see the impact of the activation function, and observe the effects of noise and batch size.

[Create a COCO dataset from scratch](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch)

Questions:

- What if we don't include an activation layer like ReLU?
- What is Softmax?
- What is the 'wd' parameter? Weight Decay is a hyperparameter, as explained in this [article](https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab).


Tasks:

- Train a segmentation model on the COCO dataset.
- Create a Yin & Yang model.

## [2020-04-12]({{< ref "/days/ml#2020-04-12" >}}) {#2020-04-12}

Continuing learning and training models.

I've read an [article that summarizes the fastai course](https://towardsdatascience.com/10-new-things-i-learnt-from-fast-ai-v3-4d79c1f07e33).

Datasets:

- Train: Used to train the model.
- Validation: Provides an unbiased evaluation while tuning model hyperparameters.
- Test: Used when the model is fully trained.

In NLP, the first step is training a language model, which is the step of guessing the next word in a sentence.

A language model has its own labels.

Now we can create a classifier.

MSE: Mean Squared Error.
Always non-negative.

Article to understand and choose the last layer activation and loss function.

Questions:

- What is regularization L1, L2?
- What is momentum?

Tasks:

- Train a model for NLP sentiment analysis.
- Recreate MSE function.


## [2020-04-13]({{< ref "/days/ml#2020-04-13" >}}) {#2020-04-13}

I've explored the batch size and tried to understand how it works. It's related to PyTorch and involves loading data per batch to avoid loading everything into memory, I guess.

I'll start lesson 5.

Fastai adds two layers at the end:

- Freeze: Freezes the previous layers.
- Split the layers into 2 groups with different learning rates.
- Discriminative learning rate.

```python
learn.fit(n_epoch, learning_rate)

# Learning rate parameter
# All layers receive the same learning rate
1e-3

# The final layer receives the indicated learning rate. The other layers receive the number divided by 3.
slice(1e-3)

# The first layer receives the first value,
# All intermediate layers receive multiplicatively equal spreads,
# The last layer receives the second value.
slice(1e-5, 1e-3)
```

Add learning rate per group, not per layer.

`cnn_learner` has 3 layer groups by default.

Pre-processing:

- [One hot encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) converts input to 0 or 1.
- Matrix product by a one-hot encoding matrix, similar to array lookup, is called embedding.
- `y_range` makes the final activation function a sigmoid.
- RMSE^2 → MSE
- PCA: Principal Component Analysis
- `parameters -= learning_rate * parameters.grad`

> *Matrix multiplications followed by ReLUs, when stacked together, have this amazing mathematical property known as the universal approximation theorem. If you have large enough weight matrices and enough of them, they can solve any arbitrarily complex mathematical function to any arbitrarily high level of accuracy.*

Fine-tuning: Based on a trained model, retrain the model to fit our use case.

Tasks:

- Write a linear function.
- Calculate RMSE.

Questions:

Why use square instead of absolute in the loss function?

Okay, it’s called MAE, and it’s used when we don’t care about outliers.

[Loss functions: MSE, MAE, Huber](https://towardsdatascience.com/understanding-the-3-most-common-loss-functions-for-machine-learning-regression-23e0ef3e14d3)

Another loss function that exists is named Huber.


## [2020-04-15]({{< ref "/days/ml#2020-04-15" >}}) {#2020-04-15}

I want to write the MSE, MAE, and RMSE functions.

Clarifying the squaring and square rooting.

Squared:

$$
x^2
$$

Square root:

$$
\sqrt{x}
$$

Here are the Python functions for MSE, RMSE, and MAE:

```python
# Mean Squared Error (MSE)
def mse(target, prediction):
    return ((target - prediction) ** 2).mean()

# Root Mean Squared Error (RMSE)
def rmse(target, prediction):
    return math.sqrt(mse(target, prediction))

# Mean Absolute Error (MAE)
def mae(target, prediction):
    return abs(target - prediction).mean()
```

These functions calculate the respective errors between the target values and the predictions.


## [2020-04-16]({{< ref "/days/ml#2020-04-16" >}}) {#2020-04-16}

I'll be focusing on writing a neural net from scratch.

I have started to rewrite the SGD (Stochastic Gradient Descent) functions. I'm following this article: [https://www.charlesbordet.com/fr/gradient-descent/#et-les-maths-dans-tout-ca-](https://www.charlesbordet.com/fr/gradient-descent/#et-les-maths-dans-tout-ca-). It's well explained.

Questions:

- What are affine functions?
- How do you calculate the derivative?

## [2020-04-17]({{< ref "/days/ml#2020-04-17" >}}) {#2020-04-17}

Google provides a course with a clear path and good explanations:

[Introduction to Machine Learning](https://developers.google.com/machine-learning/crash-course/ml-intro?hl=en)


## [2020-04-18]({{< ref "/days/ml#2020-04-18" >}}) {#2020-04-18}

Working with the basic implementation of the functions.

[Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)

Weight Decay:

We want a lot of parameters but less complexity.

How to penalize complexity:

Sum up the values of the parameters.

Some parameters are negative and others positive, so we can sum the squares of the parameters.

The problem with that is the number is too big, and the best loss would be to set all the parameters to zero.

So, let's multiply that with a number we choose. That number is called wd → weight decay.

A good wd is 0.1; the default is 0.01.

I have rewritten a matrix multiplication function.

The gradient is like the delta between two values.

The gradient of an array of values will be the number needed to reach the next one.

From 1 to 1.5, the gradient will be 0.5.

```python
np.gradient([1, 1.5])

# Output: array([0.5, 0.5])
```

Derivative:

$$
\frac{d}{dx}
$$

$$
f'
$$

Notation of the Gradient of a function:

$$
\vec \nabla f
$$

To calculate the gradient, we are based on the previous weights.

Based on the weight of the previous epoch minus the learning rate multiplied by the derivative of the loss function.

$$
wt = w_{t-1} - lr \times \dfrac{dL}{dw_{t-1}}
$$

dL → Loss

$$
L(x,w) = mse(m(x,y), y) + wd \times \sum w^2
$$

A gradient is a vector; a directional derivative is a scalar.

Derivative based on the gradient gets the direction to choose to decrease the cost.

An implementation of gradient descent in python: [gradient_descent.py](https://gist.github.com/ajmaradiaga/118f55ef4999318d6640232a73a735bd)

Questions:

- How to compute SGD on all the parameters?
- How to calculate the gradient of a tensor?

## [2020-04-19]({{< ref "/days/ml#2020-04-19" >}}) {#2020-04-19}

Discussion with Natan about Fastai and the multi-label training on the Planet dataset.

`lr_find()` restores the weights at the end of the exploration.

The threshold should not impact the training, but what if we set the threshold to 1%? Should we get an accuracy of 100%?

To know the loss function used, use `learn.loss_func`.

To launch the debugger in Jupyter, use `%debug`.

[Tutorial to implement a NN from scratch](https://pytorch.org/tutorials/beginner/nn_tutorial.html).

Logistic regression model → Neural net with one layer and no hidden layer.

CrossEntropyLoss is used for problems where the loss can't be calculated between elements.
If we predict 5 instead of 4, the error is the same as predicting 0 instead of 4.

There's no relation between the numbers; we can't calculate the loss between 5 and 4.

Gradient of 

$$
\frac{d\ (wd \times w^2)}{dw} = 2 \times wd \times w
$$

where wd → constant, w → weights.

➖ **Weight Decay:** Subtracts some constant times the weights every time we do a batch.


$$
wd \times w
$$

➕ **L2 Regularization:** Adds the square of the weights to the loss function.


$$
wd \times w^2
$$

Momentum:\
Exponentially weighted moving average:

$$
S_t = \beta \times g_t + (1 - \beta) \times S_{t-1}
$$

where \( S_t \) is the step at time t, \( g_t \) is the gradient at time t, and \( \beta \) is the momentum coefficient.

SGD with Momentum → Based on the current gradient plus the exponentially weighted moving average of the last few steps.\
SGD Momentum is almost always set to 0.9.

RMSPROP:\
If the gradient is consistently very small and not volatile, make bigger jumps.

ADAM:\
Combines RMSPROP and Momentum.

Cross-Entropy Loss:\
Sum of the one-hot encoded variables times all the activations.

Can be solved with an if statement:\
if cat then log(cat_pred) else log(1 - cat_pred).

Can be done by a lookup:\
Look up the log of the activation for the correct answer.\
Ensure the prediction sums up to one.\
Using softmax in the last layer: All activations are greater than 0 and less than 1.

For single-label multi-class classification, use softmax as the activation function, and cross-entropy as the loss function.

Weight Decay, Dropout, and BatchNorm are regularization functions.

Tasks:

- Implement NN from scratch.
- Search for accuracy threshold.
- Write nn.Linear.

Questions:

- What is momentum?
- Why do we square in weight decay? Why not just use the absolute value?
- What is Adam optimization?
- What are the intercept and slope in a linear model?

## [2020-04-21]({{< ref "/days/ml#2020-04-21" >}}) {#2020-04-21}

Refreshing my understanding of one-hot encoding and embedding.

☝ Affine function → Linear function → Matrices Multiplication

Embedding is the use of array lookup instead of using matrix multiply by one hot encoding matrix.


## [2020-04-22]({{< ref "/days/ml#2020-04-22" >}}) {#2020-04-22}

Starting lesson 6 of fastai.

[platform.ai](http://platform.ai) To classifier unlabeled images

Add more detail about the date, the data are used as one hot embedding by the model. 

```python
add_datepart(train, "Date", drop=False)
```

## [2020-04-23]({{< ref "/days/ml#2020-04-23" >}}) {#2020-04-23}

I'm working with a notebook that utilizes GPU processing.

To switch to CPU processing:

```python
defaults.device = torch.device('cpu')
```

Fastai Preprocessing:

```python
Categorify
FillMissing
```

- `FillMissing` replaces missing values with the median.
- Handling missing values is important in a model.
- If cardinality isn't too high, convert the feature into a categorical one.

Preventing overfitting with dropout:

- Implement dropout in both training and testing phases.
- Also consider using BatchNorm.

Tasks:

- Create a function for implementing dropout. 

Note: When using dropout, it is typically applied during the training phase to prevent overfitting. During the test or evaluation phase, dropout is not used; instead, the full network is utilized.


## [2020-04-25]({{< ref "/days/ml#2020-04-25" >}}) {#2020-04-25}

I'm continuing with Lesson 6.

[Image Kernels](http://setosa.io/ev/image-kernels)

[CNNs from Different Viewpoints](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c)

Convolution involves multiplying pixel blocks with a kernel. 

A kernel can detect various features like top edges, left edges, etc.

In the case of Stride 2 convolution, every other pixel is skipped.

Average Pooling is another technique used.

With Natan, we started a project focusing on NLP. Our goal is to process our conversations on WhatsApp or Messenger and detect the sentiments within them.

I imported my WhatsApp messages and performed some cleanup.

Now, I'll focus on the first part of NLP. In Natural Language Processing, there are typically 3 parts:

1. Language Model
2. Specific Language Model
3. Classifier

I will work on the Language Model, which will be based on Wikipedia content. Since we want to analyze conversations in French, the language model needs to be in French.

The language model's task is to predict the next word in a sentence.
There's an existing Language Model created from Wikipedia: https://github.com/piegu/language-models/blob/master/lm2-french.ipynb 

In NLP, there are Bidirectional Language Models where predictions can be made both **backward** and **forward**.

I'm figuring out how to load this model into Fastai. I have the weights and need to create a `language_model_learner` with them. It seems I need the data - to predict the next word, the model needs something to refer to.

The vocab appears to be included in the files, so I'm figuring out how to load the weights and vocab without needing the entire dataset, and still be able to use `learn.predict`. I might need a corpus; initially, I'll try using the one from Wikipedia.


## [2020-04-26]({{< ref "/days/ml#2020-04-26" >}}) {#2020-04-26}

Continuing with the NLP project, I'm researching how NLP is implemented in Fastai, aiming to customize some parts of the workflow with our custom data.

Fastai discusses MultiFiT for text classification, a method based on ULMFiT.

https://nlp.fast.ai/

For our NLP project, we scraped a French website for cinema reviews.


## [2020-04-29]({{< ref "/days/ml#2020-04-29" >}}) {#2020-04-29}

I continue working on our NLP project. There was an issue with the data at some point; I'll check why and try to fix the problem I'm encountering with the training of the language model.

- [QRNN: Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576)


## [2020-04-30]({{< ref "/days/ml#2020-04-30" >}}) {#2020-04-30}

I'm exploring QRNNs and trying to figure out why they're not working on the workstation.

## NeXT

To follow the upcoming days:

{{< subscribe html >}}
{{< js >}}




