---
title: "Learning Machine Learning"
date: 2022-08-02T00:00:00Z
draft: false
type: "day_to_day"
number_of_days: 9
updated_on: false
---

Machine Learning and Deep Learning are trendy topics and it is a promising area. As a Software Engineer, it's identified as a Data Scientist subject but it sounded so cool that I decided to dig it. Here are my day-to-day notes on this long journey to master the field.

---

[2020-03-17]({{< ref "/days/ml#2020-03-17" >}})  
[2020-03-18]({{< ref "/days/ml#2020-03-18" >}})  
[2020-03-19]({{< ref "/days/ml#2020-03-19" >}})  
[2020-03-20]({{< ref "/days/ml#2020-03-20" >}})  
[2020-03-21]({{< ref "/days/ml#2020-03-21" >}})  
[2020-03-22]({{< ref "/days/ml#2020-03-22" >}})  
[2020-03-23]({{< ref "/days/ml#2020-03-23" >}})  
[2020-03-24]({{< ref "/days/ml#2020-03-24" >}})  
[2020-03-25]({{< ref "/days/ml#2020-03-25" >}})  

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


## NeXT

To follow the upcoming days:

{{< subscribe html >}}
{{< js >}}




