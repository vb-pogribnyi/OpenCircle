# Open Circle

## Introduction

This article is an attempt to describe a machine learning model performance for a simple type of images. The images are generated synthetically, multiple model architectures with different parameters are tested. Afterwards, I try to explain the performance principles for each trained model. 

## 1. Dataset

The dataset consists of images of open circles 30x30 pixels.

We use bezier lines for curved parts

We will leave open part on the circle at some angle. The label will be sin and cos of that angle, because it will make easier to set the objective for the training.

The image itself is drawn using OpenCV

Noise is added ontop of the image both as a white noise and line disturbances

## 2. Model

We will test several architectures:

- Two large kernels (5-7 px), large pooling (2-4 px), dense
- Two small kernels (3 px), large kernel (5 px) 4 channels, 2 px pooling between them, dense layer
- Two small kernels (3 px), large kernel 5 px 1 channel, flipped vertically and horizontally to obtain 4 outputs, dense layer
- One large kernel (5-7 px), large kernel 5 px 1 channel, flipped vertically and horizontally to obtain 4 outputs, dense layer

For each architecture, we will launch multiple experiments with different number of channels, to figure out which layers are important.

In this article we will describe the process for the following architecture:

- Conv 3x3, 1 channel
- Tanh activation
- Avg pooling
- 
- Conv 3x3, 1 channel
- Tanh activation
- Avg pooling
- 
- Conv 5x5, 4 channels
- Tanh activation
- Avg pooling
- 
- Dense layer, 4x2
- Tanh activation
- Dense layer 2x2
- Tanh activation

The output (2 values) corresponds to sin and cos of the target angle.

## 3. Training

The models are trained with 256k examples with no regularization, using Adam optimizer.

## 4. Analysis

### 4.1 Dense layer

The best result is obtained with 4-2-2 dense network architecture. 

Print out dependencies between the hidden layer and output, then the input and hidden layer (the last one is 4 dimensional, which is not a problem)

Here's how the last layers looks like:

![dense2](C:\Users\vpogribnyi\Pictures\OpenCircle\dense2.png)

Next I will assume that the hidden state is given by two numbers from -1 to +1, i.e. (0.3, -0.8), since it has two neurons and previous layer activation function is tanh.

So, what does the image above say us. First of all, at the input (0, 0) the output will be (0, 0) as well. Actually it will be (0, 0) plus bias, but we are not concerned by the bias for now.

Let's go further. Input (1, 1) would yield approximately (0, -1), which is, given these are sin and cos of the angle, 180°. Input (-1, -1) would be (0, 1) which is 0°. You should start getting the idea by now. 

Let's now take a look at the first dense layer:

![dense1_0](C:\Users\vpogribnyi\Pictures\OpenCircle\dense1_0.png)

This plot represents 4 dimensional space, so it looks a bit complicated. To simplify things a little bit, we first examine a half of the layer. Only a part which takes 4 values after the CNN and outputs the value for the first hidden neuron. This part is represented by the image above, I have another one similar for the second hidden neuron. Let's take a look. We got a grid 5x5 images, each is a plot itself, similar to the previous image. Each image shows a dependency of the output on the first two inputs, given the last two inputs. Let me explain by example. We have four values as input. Let's call them [v1, v2, v3, v4]. And let's call our output 'o'. So the first plot (top left) shows dependency o(v1, v2) given v3 = -1 and v4 = -1. Top right plot shows o(v1, v2) given v3 = 1 and v4 = -1. And so on: middle plot shows o(v1, v2) given v3 = 0 and v4 = 0; bottom left shows o(v1, v2) given v3 = -1 and v4 = 1; bottom right shows o(v1, v2) given v3 = 1 and v4 = 1.

So what do we see on these plots. First, for v3 = 1 and v4 = 1, as well as for v3 = -1 and v4 = -1 - no matter what are v1 and v2, the output will be close to -1 or +1. The activation function goes saturated. At other points the layer shows the dependency similar to the previous plot.

Finally, let's connect the four inputs to the sin out. Let's start easy and examine first the case when  for v3 = 1 and v4 = 1. Our output is close to +1, which means the input to the second layer is (+1, ?), since the second neuron value is unknown. We did not get to it yet. If we take a look at the previous plot, our sin value will range in values [0, 1].

### 4.2 Convolutions

To get an idea of how the convolutions work, we will print out its outputs for different inputs, and try to understand its operation visually.

For simplicity, we will not analyze how the first two convolution layers work, and focus on the third one. There are four filters in the convolution. To start with, let's take a sample input into the layer, and look at how it will be processed by the first filter. Take a look at this visualization:

![40_0.6_0](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v2\data\40_0.6_0.png)

We see a bunch of things going on here, let's take a look. First, the input to the convolution is 6x6, the filter is 5x5. Consequently the output is 2x2. The input is labeled here as "Original", filter - as "Filter". Images "Out 1" - "Out 4" shows the filter multiplied by different parts of the image (the thing that convolution does). The remainder of the output labels prints the result of applying the filter, bias, and final result. So the formula is as follows: "<weight * image slice> + <bias> = <final output>"

Second, the network part illustrated here includes: input -> tanh activation -> convolution -> average pooling. The input is marked "Original", as I mentioned, input after the tanh marked "Src", output before pooling marked "Output", output after pooling is listed in the first (text) cell, labeled "Grand output". The text cell also contains filter index (0; varies in range 0-3). Don't worry about "Example" and "Grand index", we're getting to it.

Well, the image above describes how a single filter works, but doesn't tell a lot about the layer operation. To get this idea, let's collect a bunch of inputs and pass them through the convolution. Then arrange the outputs in ascending order, and print out the images, so that we see what kind of input will give a particular output. 

For example, following is output from the first convolution for a bunch of inputs:

![img_0_0.3](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v2\data\img_0_0.3.png)

You may have noticed a couple things about this image. First, there are lot of inputs, each is a circle opened at the bottom. Only image position and noise varies here. Second, there are numbers above each image, formatted like "74_1.0". Here 74 is something called "Example index". If I want to see other filter's output for this same input, I will find it by this id. The second part, 1.0 is the convolution output for this input, multiplied by 5, and rounded. I decided to multiply the output by 5, so that it would be easier to range. 

To make it clear, let's take a look at the first (text) cell on the previous image. We have there "Example" - example index; "Grand output" - output after pooling; "Grand index" - grand output multiplied by 5. For our particular example it is 1.84. If rounded, this number will be 2.0. We can find this input on the last image by its example index: it's labeled "40_2.0"