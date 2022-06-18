---
title: "Learning Machine Learning"
date: 2022-06-15T00:00:00Z
draft: false
type: "day_to_day"
number_of_days: 3
updated_on: false
---

Machine Learning and Deep Learning are trendy topics and it is a promising area. As a Software Engineer, it's identified as a Data Scientist subject but it sounded so cool that I decided to dig it. Here are my day-to-day notes on this long journey to master the field.

---

[2020-03-17]({{< ref "/days/ml#2020-03-17" >}})  
[2020-03-18]({{< ref "/days/ml#2020-03-18" >}})  
[2020-03-19]({{< ref "/days/ml#2020-03-19" >}})  

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


## NeXT

To follow the upcoming days:

{{< subscribe html >}}
{{< js >}}




