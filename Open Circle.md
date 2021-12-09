# Open Circle

## Introduction

This article is an attempt to describe a machine learning model performance for a simple type of images. The images are generated synthetically, multiple model architectures with different parameters are tested. Afterwards, I try to explain the performance principles for each trained model. 

## 1. Dataset

The dataset consists of images of open circles 30x30 pixels. There will be something I call 'core image', which is a collection of x-y coordinates of points, meant to be connected by lines. This core image will be then drawn on a bitmap, a noise will be added to it, and it will be fed to the NN model.

The core image will represent a circle itself. But since I want to generate all kinds of circle variations, including ovals, circles with linear edges (half circle - half rectangle) - I will need to introduce some additional logic. After the image is generated - it will be scaled and rotated at a random angle, because, again, we want to generate all kinds of circles.

I guess the introduction was not really clear about what we're going to do. I will explain all the details below, and include examples. So keep up!

Code described in this chapter is assumed to be kept in a single file called 'generate_dataset.py'

### 1.1 Bezier curves

So, our core image consists of curved parts and linear parts. The curved parts may be modeled different ways, but I prefer bezier curves, because of their simplicity and flexibility. The curve will be built on three points. The first point will be (0, 0), two other will define at what points our circle touches edges. 

For example, if I want to build a circle with no linear parts, I will choose these points:

```python
point1 = {'x': 0, 'y': 0.5}
point2 = {'x': 0, 'y': 0}
point3 = {'x': 0.5, 'y': 0}
```

If I build a bezier curve on these points, I will get something like this:

![curve_1](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\curve_1.png)

The code so far looks as follows:

```python
import numpy as np
import matplotlib.pyplot as plt

def get_corner_points(n):
    point1 = {'x': 0, 'y': 0.5}
    point2 = {'x': 0, 'y': 0}
    point3 = {'x': 0.5, 'y': 0}

    xs = np.zeros(n)
    ys = np.zeros(n)
    for i, t in enumerate(np.linspace(0, 1, n)):
        xs[i] = (1-t)**2 * point1['x'] + \
                2*(1-t)*t * point2['x'] + \
                (t)**2 * point3['x']
        ys[i] = (1-t)**2 * point1['y'] + \
                2*(1-t)*t * point2['y'] + \
                (t)**2 * point3['y']
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

if __name__ == '__main__':
    xs, ys = get_corner_points(100)
    plt.plot(xs, ys)
    plt.gcf().set_size_inches(5, 5)
    plt.show()
```

The get_corner_points function builds the curve itself. It accepts the number of points generated, as parameter. The function is called this way because it generates 'corners' of our circle. If I mirror them around the image, I will get a complete circle. In other words, if I changed the main code this way:

```python
if __name__ == '__main__':
    xs, ys = get_corner_points(100)
    plt.plot(xs, ys)
    plt.plot(1 - xs, ys)
    plt.plot(xs, 1 - ys)
    plt.plot(1 - xs, 1 - ys)
    plt.gcf().set_size_inches(5, 5)
    plt.show()
```

I will get this kind of circle:![curve_2](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\curve_2.png)

Notice that our circle touches axis at points (0, 0.5), (0.5, 0). So if I change my bezier points to something like this:

```python
point1 = {'x': 0, 'y': 0.4}
point2 = {'x': 0, 'y': 0}
point3 = {'x': 0.4, 'y': 0}
```

I will get the following:

![curve_3](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\curve_3.png)

Now if  I want to get a circle with flat sides, I have to generate an image like this and connect the gaps. But before that, let's add a minor change to the existing code. Let's add another two parameters to the get_corner_points(), so that we can generate circles with larger or smaller flat sides:

```python
def get_corner_points(p1, p2, n):
    point1 = {'x': 0, 'y': 0.5 * p1}
    point2 = {'x': 0, 'y': 0}
    point3 = {'x': 0.5 * p2, 'y': 0}
```

Parameters p1 and p2 will vary in range (0, 1) - giving a square at 0 and a circle at 1. The example when p1 = 0.4 and p2 = 0.8 looks like this:

![curve_4](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\curve_4.png)

The full code so far is like this:

```python
import numpy as np
import matplotlib.pyplot as plt

def get_corner_points(p1, p2, n):
    point1 = {'x': 0, 'y': 0.5 * p1}
    point2 = {'x': 0, 'y': 0}
    point3 = {'x': 0.5 * p2, 'y': 0}

    xs = np.zeros(n)
    ys = np.zeros(n)
    for i, t in enumerate(np.linspace(0, 1, n)):
        xs[i] = (1-t)**2 * point1['x'] + \
                2*(1-t)*t * point2['x'] + \
                (t)**2 * point3['x']
        ys[i] = (1-t)**2 * point1['y'] + \
                2*(1-t)*t * point2['y'] + \
                (t)**2 * point3['y']
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

if __name__ == '__main__':
    xs, ys = get_corner_points(0.4, 0.8, 100)
    plt.plot(xs, ys)
    plt.plot(1 - xs, ys)
    plt.plot(xs, 1 - ys)
    plt.plot(1 - xs, 1 - ys)
    plt.gcf().set_size_inches(5, 5)
    plt.show()
```

### 1.2 Full circle

After the corner parts are created, they should be connected to complete the circle. What's more, we want all the points to be distributed uniformly (which means that the distance between two neighboring points is roughly the same over all the images).

So, first things first. Let's create a function that will generate our circle. It will accept the same parameters as the get_corner_points(). The third parameter 'n' will be the number of points in the (whole) resulting circle.

```python
def get_circle(p1, p2, n):
    corner1_xs, corner1_ys = get_corner_points(p1, p2, n)
    corner2_xs, corner2_ys = 1 - corner1_xs, corner1_ys
    corner3_xs, corner3_ys = corner1_xs, 1 - corner1_ys
    corner4_xs, corner4_ys = 1 - corner1_xs, 1 - corner1_ys
```

 This is exactly the same as the main function from the previous chapter. But there is an issue with points ordering. Let's draw these corners again, but also mark the first and last points for each corner:

 ![curve_5](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\circle_5.png)

The image on the right shows what we have now. Notice that if we follow the order of the points counter-clockwise, we start at corner 4, first point, then meet corner 4 last point, then corner 3 last point, corner 3 first point, corner 1 first point, and so on. I want to reorder things around so they look like on the image on the left:

```python
def get_circle(p1, p2, n):
    corner3_xs, corner3_ys = get_corner_points(p1, p2, n)
    corner4_xs, corner4_ys = 1 - corner3_xs[::-1], \
                             corner3_ys[::-1]
    corner2_xs, corner2_ys = corner3_xs[::-1], \
                             1 - corner3_ys[::-1]
    corner1_xs, corner1_ys = 1 - corner3_xs, 1 - corner3_ys
```

Now adding the lines between the corners is not a problem:

```python
    line12_y = np.ones(n)
    line12_x = np.linspace(corner1_xs[-1],
                           corner2_xs[0], n + 2)[1:-1]
    line23_y = np.linspace(corner2_ys[-1],
                           corner3_ys[0], n + 2)[1:-1]
    line23_x = np.zeros(n)
    line34_y = np.zeros(n)
    line34_x = np.linspace(corner3_xs[-1],
                           corner4_xs[0], n + 2)[1:-1]
    line41_y = np.linspace(corner4_ys[-1],
                           corner1_ys[0], n + 2)[1:-1]
    line41_x = np.ones(n)
```

Here line index determines what corners the line connects. For example, line12 connects corner1 and corner2; line41 connects corner4 and corner1, and so on.

Let's print the result and see what we have:

```python
    plt.plot(corner1_xs, corner1_ys)
    plt.plot(line12_x, line12_y)
    plt.plot(corner2_xs, corner2_ys)
    plt.plot(line23_x, line23_y)
    plt.plot(corner3_xs, corner3_ys)
    plt.plot(line34_x, line34_y)
    plt.plot(corner4_xs, corner4_ys)
    plt.plot(line41_x, line41_y)
    plt.gcf().set_size_inches(5, 5)
    plt.show()
```

![circle_6](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\circle_6.png)

Looks fine. Now let's collect all the points in one array, by concatenation. There is only one issue. I want the array's first point to be the one at 'zero degrees', or this one:

![circle_7](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\circle_7.png)

This will make me to divide line41 into halves, then concatenate the points:

```python
    xs = np.concatenate([
        line41_x[len(line41_x) // 2:],
        corner1_xs, line12_x,
        corner2_xs, line23_x,
        corner3_xs, line34_x,
        corner4_xs, line41_x[:len(line41_x) // 2]
    ])
    ys = np.concatenate([
        line41_y[len(line41_y) // 2:],
        corner1_ys, line12_y,
        corner2_ys, line23_y,
        corner3_ys, line34_y,
        corner4_ys, line41_y[:len(line41_y) // 2]
    ])
```

Now by plotting the xs and ys, and by plotting them partially, we get our open circle core image:

```python
    plt.subplot(1, 2, 1)
    plt.plot(xs, ys)
    plt.subplot(1, 2, 2)
    plt.plot(xs[15:-15], ys[15:-15])
    plt.gcf().set_size_inches(10, 5)
    plt.show()
```

![circle_8](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\circle_8.png)

Now we need to make sure that the distance between the points is uniform. The easy way to do it would be to use numpy interp function. To do that, I will create an array of distances for the current image, then an array of desired (uniform) distances, and interpolate the xs and ys arrays on the desired distances:

```python
    dx = np.diff(xs)
    dy = np.diff(ys)
    distances = np.square(dx ** 2 + dy ** 2)
    location_old = [0]
    for d in distances:
        location_old.append(location_old[-1] + d)
    location_old = np.array(location_old)
    location_old /= np.sum(distances)
    location_new = np.linspace(0, 1, n)
    xs = np.interp(location_new, location_old, xs)
    ys = np.interp(location_new, location_old, ys)
```

Now xs and ys arrays have n items each, and if we plot the images above again, we get:

![circle_9](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\circle_9.png)

Notice that image on the left is cut by 30% (since we do this line):

```python
plt.plot(xs[15:-15], ys[15:-15])
```

Now I will add to this function only a return value. The final code looks like this (assuming the code from the previous chapter is still at place):

```python
def get_circle(p1, p2, n):
    corner3_xs, corner3_ys = get_corner_points(p1, p2, n)
    corner4_xs, corner4_ys = 1 - corner3_xs[::-1], \
                             corner3_ys[::-1]
    corner2_xs, corner2_ys = corner3_xs[::-1], \
                             1 - corner3_ys[::-1]
    corner1_xs, corner1_ys = 1 - corner3_xs, 1 - corner3_ys
    line12_y = np.ones(n)
    line12_x = np.linspace(corner1_xs[-1],
                           corner2_xs[0], n + 2)[1:-1]
    line23_y = np.linspace(corner2_ys[-1],
                           corner3_ys[0], n + 2)[1:-1]
    line23_x = np.zeros(n)
    line34_y = np.zeros(n)
    line34_x = np.linspace(corner3_xs[-1],
                           corner4_xs[0], n + 2)[1:-1]
    line41_y = np.linspace(corner4_ys[-1],
                           corner1_ys[0], n + 2)[1:-1]
    line41_x = np.ones(n)

    xs = np.concatenate([
        line41_x[len(line41_x) // 2:],
        corner1_xs, line12_x,
        corner2_xs, line23_x,
        corner3_xs, line34_x,
        corner4_xs, line41_x[:len(line41_x) // 2]
    ])
    ys = np.concatenate([
        line41_y[len(line41_y) // 2:],
        corner1_ys, line12_y,
        corner2_ys, line23_y,
        corner3_ys, line34_y,
        corner4_ys, line41_y[:len(line41_y) // 2]
    ])

    dx = np.diff(xs)
    dy = np.diff(ys)
    distances = np.square(dx ** 2 + dy ** 2)
    location_old = [0]
    for d in distances:
        location_old.append(location_old[-1] + d)
    location_old = np.array(location_old)
    location_old /= np.sum(distances)
    location_new = np.linspace(0, 1, n)
    xs = np.interp(location_new, location_old, xs)
    ys = np.interp(location_new, location_old, ys)

    return xs, ys


if __name__ == '__main__':
    xs, ys = get_circle(0.4, 0.8, 100)
    plt.subplot(1, 2, 1)
    plt.plot(xs, ys)
    plt.subplot(1, 2, 2)
    plt.plot(xs[15:-15], ys[15:-15])
    plt.gcf().set_size_inches(10, 5)
    plt.show()
```

### 1.3 Random transform

To generate the 'all kinds of open circles', let's rotate and scale what we have. The operation will be done by matrix multiplication. The function responsible for this transform will be accepting the following parameters:

- x and y coordinates of our circle
- rotation angle
- scale x, scale y

Let's create a stub for the function:

```python
def transform(xs, ys, theta, xscale, yscale):
    for i in range(len(xs)):
        pass # Do matrix multiplication
    return xs, ys
```

To rotate a point around origin of coordinates, we have to multiply it by the following matrix (see Wikipedia for rotation matrix):
$$
\left[ 
\begin{array}{c | c} 
  \begin{array}{c c c} 
     \hat{x}\\ 
     \hat{y}
  \end{array} \\ 
 \end{array} 
\right] = \left[ 
\begin{array}{c | c} 
  \begin{array}{c c c} 
     cos\theta & -sin\theta\\ 
     sin\theta & cos\theta
  \end{array} \\ 
 \end{array} 
\right]\left[ 
\begin{array}{c | c} 
  \begin{array}{c c c} 
     x\\ 
     y
  \end{array} \\ 
 \end{array} 
\right]
$$
There are two issues for now: first, we need to move the center of our circle to the origin of coordinates. Second, our theta has to be in radians. Let's fix these in code, and add the matrix coefficients:

```python
def transform(xs, ys, theta, xscale, yscale):
    # Move to origin of coordinates
    xs -= 0.5
    ys -= 0.5
    # Convert theta to rad
    theta = theta / 180 * np.pi
    a11 = np.cos(theta)
    a12 = -np.sin(theta)
    a21 = np.sin(theta)
    a22 = np.cos(theta)
    for i in range(len(xs)):
        pass # Do matrix multiplication
    return xs, ys
```

Now we only have to do actual scaling and matrix multiplication. Let's add these to the code:

```python
def transform(xs, ys, theta, xscale, yscale):
    # Move to origin of coordinates
    xs -= 0.5
    ys -= 0.5
    # Convert theta to rad
    theta = theta / 180 * np.pi

    xs *= xscale
    ys *= yscale
    a11 = np.cos(theta)
    a12 = -np.sin(theta)
    a21 = np.sin(theta)
    a22 = np.cos(theta)
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        xs[i] = x * a11 + y * a12
        ys[i] = x * a21 + y * a22
    return xs, ys
```

Also don't forget to return the image back from the center of the origin. The full function along with the code calling it looks like this:

```python
def transform(xs, ys, theta, xscale, yscale):
    # Move to origin of coordinates
    xs -= 0.5
    ys -= 0.5
    # Convert theta to rad
    theta = theta / 180 * np.pi

    xs *= xscale
    ys *= yscale
    a11 = np.cos(theta)
    a12 = -np.sin(theta)
    a21 = np.sin(theta)
    a22 = np.cos(theta)
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        xs[i] = x * a11 + y * a12
        ys[i] = x * a21 + y * a22
    xs += 0.5
    ys += 0.5
    
    return xs, ys


if __name__ == '__main__':
    xs, ys = get_circle(0.4, 0.8, 100)
    xs, ys = transform(xs, ys, 10, 0.8, 1.3)

    plt.subplot(1, 2, 1)
    plt.plot(xs, ys)
    plt.subplot(1, 2, 2)
    plt.plot(xs[15:-15], ys[15:-15])
    plt.gcf().set_size_inches(10, 5)
    plt.show()
```

If we run the code, we get the following image:

![circle_10](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\circle_10.png)

Now we're ready to move to the next stage - draw an actual image, with pixels - using OpenCV.

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