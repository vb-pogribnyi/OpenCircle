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

### 1.4 Converting to pixels

The function we're about to write will be accepting an array of point coordinates (which we already have), and output our final 30x30 image. The function will draw the points as lines, and will use OpenCV library for that. So the first thing is, obviously, to import OpenCV to our project:

```python
import cv2 as cv
```

Next let's make our function stub and change main function so that we can see the result:

```python
def to_image(xs, ys, img_size):
    result = np.zeros((img_size, img_size))

    return result


if __name__ == '__main__':
    xs, ys = get_circle(0.4, 0.8, 100)
    xs, ys = transform(xs, ys, 10, 0.8, 1.3)
    img = to_image(xs, ys, 30)

    plt.pcolor(img, cmap='Wistia')
    plt.gcf().set_size_inches(5, 5)
    plt.show()
```

Now let's add drawing procedure. Keep in mind that our core image is in range (0, 1) - this should be mapped to (0, img_size), otherwise our image will appear as a single dot:

```python
def to_image(xs, ys, img_size):
    result = np.zeros((img_size, img_size))
    xs = (xs * img_size).astype(int)
    ys = (ys * img_size).astype(int)
    thickness = 1
    for i in range(1, len(xs)):
        cv.line(result, 
                (xs[i - 1], ys[i - 1]), 
                (xs[i], ys[i]), 1, thickness)

    return result
```

After running the code, we get the following image:

![image_1](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\image_1.png)

Notice that the parts where our core image has gone below zero or above one - are not drawn. We can draw another image with smaller scale values, so that it fits:

```python
xs, ys = transform(xs, ys, 10, 0.8, 0.6)
```

And get this:![image_2](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\image_2.png)

Yep, this is how our image looks like. Let's add noise to it, since we know that we need noise in order to train a neural network. We will be adding multiple kind of noises. First is vertical/horizontal shift, so that all points are shifted at random value:

```python
    # Add shift noise
    shift_x = np.random.uniform(-0.1, 0.1)
    shift_y = np.random.uniform(-0.1, 0.1)
    xs += shift_x
    ys += shift_y
```

Second is noise added to every point:

```python
    # Every point noise
    img_noise_std = np.random.uniform(0.01, 0.03)
    xs += np.random.normal(0, img_noise_std, xs.shape)
    ys += np.random.normal(0, img_noise_std, ys.shape)
```

These types of noise are applied before the image is scaled. Third type of noise will be regular white noise, and will be applied to the whole image after it is drawn:

```python
# White noise
white_noise_std = np.random.uniform(0.1, 0.3)
result += np.random.normal(0, white_noise_std, result.shape)
```

Here is the full function code:

```python
def to_image(xs, ys, img_size):
    result = np.zeros((img_size, img_size))

    # Add shift noise
    shift_x = np.random.uniform(-0.1, 0.1)
    shift_y = np.random.uniform(-0.1, 0.1)
    xs += shift_x
    ys += shift_y

    # Every point noise
    img_noise_std = np.random.uniform(0.01, 0.03)
    xs += np.random.normal(0, img_noise_std, xs.shape)
    ys += np.random.normal(0, img_noise_std, ys.shape)

    xs = (xs * img_size).astype(int)
    ys = (ys * img_size).astype(int)

    thickness = 1
    for i in range(1, len(xs)):
        cv.line(result,
                (xs[i - 1], ys[i - 1]),
                (xs[i], ys[i]), 1, thickness)
    # White noise
    white_noise_std = np.random.uniform(0.1, 0.3)
    result += np.random.normal(0, white_noise_std, result.shape)


    return result


if __name__ == '__main__':
    xs, ys = get_circle(0.4, 0.8, 100)
    xs, ys = transform(xs, ys, 10, 0.8, 0.6)
    img = to_image(xs, ys, 30)

    plt.pcolor(img, cmap='Wistia')
    plt.gcf().set_size_inches(5, 5)
    plt.show()
```

And here is our output so far:

![image_3](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\image_3.png)

This looks fine, but I'd like to add a couple more details. First, blur. I'd like to have the core image a little blurred, then noise a little blurred also, then add not blurred noise ontop. Here's what it looks like in code:

```python
    # Blur image and noise
    blur_noise_std = np.random.uniform(0.1, 0.3)
    blur_noise = np.random.normal(0, blur_noise_std, result.shape)
    img_blur = np.random.randint(1, 3)
    noise_blur = np.random.randint(1, 5)
    cv.blur(result, (img_blur, img_blur), result)
    cv.blur(blur_noise, (noise_blur, noise_blur), blur_noise)
    result += blur_noise
```

Second, line thickness. I'd like to randomize it as well. This is an easy change, since we use 'thickness' in cv.line() call. I just have to initialize the thickness to a random value:

```python
thickness = np.random.randint(1, 4)
```

Here's what a sample output looks like:

![image_4](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\image_4.png)

Now the last part. Before feeding this image into a model, we have to make sure that its values lie in certain range (0 to 1). To do that, I will add the following lines:

```python
    result -= result.min()
    result /= result.max()
```

That's it for the image. Here's what the full function code looks like:

```python
def to_image(xs, ys, img_size):
    result = np.zeros((img_size, img_size))

    # Add shift noise
    shift_x = np.random.uniform(-0.1, 0.1)
    shift_y = np.random.uniform(-0.1, 0.1)
    xs += shift_x
    ys += shift_y

    # Every point noise
    img_noise_std = np.random.uniform(0.01, 0.03)
    xs += np.random.normal(0, img_noise_std, xs.shape)
    ys += np.random.normal(0, img_noise_std, ys.shape)

    xs = (xs * img_size).astype(int)
    ys = (ys * img_size).astype(int)

    thickness = np.random.randint(1, 4)
    for i in range(1, len(xs)):
        cv.line(result,
                (xs[i - 1], ys[i - 1]),
                (xs[i], ys[i]), 1, thickness)

    # Blur image and noise
    blur_noise_std = np.random.uniform(0.1, 0.3)
    blur_noise = np.random.normal(0, blur_noise_std, result.shape)
    img_blur = np.random.randint(1, 3)
    noise_blur = np.random.randint(1, 5)
    cv.blur(result, (img_blur, img_blur), result)
    cv.blur(blur_noise, (noise_blur, noise_blur), blur_noise)
    result += blur_noise

    # White noise
    white_noise_std = np.random.uniform(0.1, 0.3)
    result += np.random.normal(0, white_noise_std, result.shape)

    result -= result.min()
    result /= result.max()
    return result
```

### 1.5 Image generator

Now that we're able to create image given some parameters, let's create a function that will generate those parameters and consequently generate the image. This function will also give actual 'opennes' to the image, and generate label for it. The function will accept no parameters, since it will generate everything, and will output the image along with its labels. The labels will be given in form of sin and cos of the angle. We don't want to use the angle itself because it will be hard for the network to figure out the importance of the error. 

For example, imagine we have a circle opened at 0.1 degrees. Our  network guesses 359 degrees, which is a good guess. But for the network the error seems high. If we use cos instead, it will be close to 1 for both 0.1 and 359 degrees.

So, the stub for our function. Basically it will contain the same functions as in the main function in previous chapter, with some changes:

```python
def generate_image():
    # Generate random values as parameters
    # for circle generation and the transform
    xs, ys = get_circle(0.4, 0.8, 100)
    xs, ys = transform(xs, ys, 10, 0.8, 0.6)
    img = to_image(xs, ys, 30)
    
    label = [0, 0] # Random label, for now
    return img, label
```

Our main function will have minor changes as well:

```python
if __name__ == '__main__':
    img, label = generate_image()

    print(label)
    plt.pcolor(img, cmap='Wistia')
    plt.gcf().set_size_inches(5, 5)
    plt.show()
```

Now we have to implement the randomizations. Let's start with the angle, so that we can have the label as well. To do that we need generate the angle in degree, then transform it into radians for the label. Then we need to insert it into transform function:

```python
    angle = np.random.uniform(0, 360)
    angle_rad = angle / 180 * np.pi
    ...
    xs, ys = transform(xs, ys, angle, 0.8, 0.6)
    ...
    label = [np.sin(angle_rad), np.cos(angle_rad)]
```

Now we add 'opennes' to the circle. This will be done by trimming some values from 'xs' and 'ys' arrays. Also, I'd like to set this value as percentage relative to whole circle length. It doesn't matter if our circle is 100 points length, but I will introduce these as parameters anyway:

```python
    n_circle_pts = 100
    open_percent = 20
    n_pts_skip = int(n_circle_pts / 100 * open_percent)
    ...
    xs, ys = get_circle(0.4, 0.8, n_circle_pts)
    ...
    img = to_image(xs[n_pts_skip // 2:-n_pts_skip // 2],
                   ys[n_pts_skip // 2:-n_pts_skip // 2], 30)
	...
```

This gives us a circle that is open in 20% of its length. Notice that if we change the number of points for the core image:

```python
n_circle_pts = 350
```

The circle will remain open in 20%.

![image_5](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\image_5.png)

Now let's randomize all the remaining parameters:

```python
def generate_image():
    n_circle_pts = 350

    # Parameters randomization
    angle = np.random.uniform(0, 360)
    open_percent = np.random.uniform(10, 40)
    circle_p1 = np.random.uniform(0.7, 1.)
    circle_p2 = np.random.uniform(0.7, 1.)
    scale_x = np.random.uniform(0.5, 1.1)
    scale_y = np.random.uniform(0.5, 1.1)

    n_pts_skip = int(n_circle_pts / 100 * open_percent)
    angle_rad = angle / 180 * np.pi
    xs, ys = get_circle(circle_p1, circle_p2, n_circle_pts)
    xs, ys = transform(xs, ys, angle, scale_x, scale_y)
    img = to_image(xs[n_pts_skip // 2:-n_pts_skip // 2],
                   ys[n_pts_skip // 2:-n_pts_skip // 2], 30)

    label = [np.sin(angle_rad), np.cos(angle_rad)]
    return img, label
```

Having updated our main function to display multiple images:

```python
if __name__ == '__main__':
    for i in range(16):
        img, label = generate_image()
        plt.subplot(4, 4, i + 1)
        plt.pcolor(img, cmap='Wistia')
    plt.gcf().set_size_inches(8, 8)
    plt.show()
```

This will give us the following sample image:

![image_6](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\image_6.png)

Looks fine to me; now we only need to save this images into a file. To do that, I will update our main function:

```python
    n_images = 1024
    images, labels = [], []
    for i in range(n_images):
        img, label = generate_image()
        images.append(img)
        labels.append(label)
    np.save('images.npy', np.array(images))
    np.save('labels.npy', np.array(labels))
```

Running this code will save to our hard drive a dataset of 1024 images. This is too little for training a  model, but good for testing. Since the process may take long for a large number of images, I will use a tqdm library:

```python
from tqdm import tqdm
```

And change my dataset generation code to this:

```python
    n_images = 256 * 1024
    images, labels = [], []
    for i in tqdm(range(n_images)):
        img, label = generate_image()
        images.append(img)
        labels.append(label)
    np.save('images.npy', np.array(images))
    np.save('labels.npy', np.array(labels))
```

Here is full dataset generation code:

```python
import cv2 as cv
import numpy as np
from tqdm import tqdm
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

def to_image(xs, ys, img_size):
    result = np.zeros((img_size, img_size))

    # Add shift noise
    shift_x = np.random.uniform(-0.1, 0.1)
    shift_y = np.random.uniform(-0.1, 0.1)
    xs += shift_x
    ys += shift_y

    # Every point noise
    img_noise_std = np.random.uniform(0.01, 0.03)
    xs += np.random.normal(0, img_noise_std, xs.shape)
    ys += np.random.normal(0, img_noise_std, ys.shape)

    xs = (xs * img_size).astype(int)
    ys = (ys * img_size).astype(int)

    thickness = np.random.randint(1, 4)
    for i in range(1, len(xs)):
        cv.line(result,
                (xs[i - 1], ys[i - 1]),
                (xs[i], ys[i]), 1, thickness)

    # Blur image and noise
    blur_noise_std = np.random.uniform(0.1, 0.3)
    blur_noise = np.random.normal(0, blur_noise_std, result.shape)
    img_blur = np.random.randint(1, 3)
    noise_blur = np.random.randint(1, 5)
    cv.blur(result, (img_blur, img_blur), result)
    cv.blur(blur_noise, (noise_blur, noise_blur), blur_noise)
    result += blur_noise

    # White noise
    white_noise_std = np.random.uniform(0.1, 0.3)
    result += np.random.normal(0, white_noise_std, result.shape)

    result -= result.min()
    result /= result.max()
    return result


def generate_image():
    n_circle_pts = 350

    # Parameters randomization
    angle = np.random.uniform(0, 360)
    open_percent = np.random.uniform(10, 40)
    circle_p1 = np.random.uniform(0.7, 1.)
    circle_p2 = np.random.uniform(0.7, 1.)
    scale_x = np.random.uniform(0.5, 1.1)
    scale_y = np.random.uniform(0.5, 1.1)

    n_pts_skip = int(n_circle_pts / 100 * open_percent)
    angle_rad = angle / 180 * np.pi
    xs, ys = get_circle(circle_p1, circle_p2, n_circle_pts)
    xs, ys = transform(xs, ys, angle, scale_x, scale_y)
    img = to_image(xs[n_pts_skip // 2:-n_pts_skip // 2],
                   ys[n_pts_skip // 2:-n_pts_skip // 2], 30)

    label = [np.sin(angle_rad), np.cos(angle_rad)]
    return img, label


if __name__ == '__main__':
    for i in range(16):
        img, label = generate_image()
        plt.subplot(4, 4, i + 1)
        plt.pcolor(img, cmap='Wistia')
    plt.gcf().set_size_inches(8, 8)
    plt.show()

    n_images = 256 * 1024
    images, labels = [], []
    for i in tqdm(range(n_images)):
        img, label = generate_image()
        images.append(img)
        labels.append(label)
    np.save('images.npy', np.array(images))
    np.save('labels.npy', np.array(labels))

```

Now if you run this on your machine, after a while you will have a dataset to train your model on.

## 2. Model

We will test two different architectures, for start (then we'll try to improve them):

- Two large kernels (5-7 px), large pooling (2-4 px), dense
- Two small kernels (3 px), large kernel (5 px) 4 channels, 2 px pooling between them, dense layer

For each architecture, we will launch multiple experiments with different number of channels, to figure out which layers are important.

### 2.1 LargeWin

LargeWin is a code name for "Two large kernels (5-7 px), large pooling (2-4 px), dense" model. In more detail, this model looks as follows:

- Conv 2d

- Tanh activation

- Avg pooling

  

- Conv 2d

- Tanh activation

- Avg pooling

  

- Flattening

- Dense layer

- Tanh activation

- Dense layer

- Tanh activation

And here is the list of parameters being varied:

- Convolutions filters number
- Convolutions kernel size
- Pooling size
- Dense network - number of hidden neurons

So here is our starting draft for the model class code:

```python
import torch

class LargeWin(torch.nn.Module):
    def __init__(self, params):
        ch1, ch2 = params['ch1'], params['ch2']
        ch3 = params['ch3']
        pool1_size = params['pool1_size']
        pool2_size = params['pool2_size']
        ks1, ks2 = params['ks1'], params['ks2']
        super(LargeWin, self).__init__()

    def forward(self, x):
        return x
```

I'm passing the parameters here as a dict, so that it will be easier to automate testing later on. Among the parameters there are: ch1, ch2 = number of filters for the convolutions. Ch3 - number of hidden neurons for the dense network. Pool1_size and pool2_size - sizes of the pooling layers after the convolutions. Ks1 and ks2 - convolutions kernel sizes. 

Let's now add the network layers. Note that dense layer input size (after flattening) will depend on the size of kernels and pooling windows. It will be calculated in the constructor as well:

```python
        self.conv1 = torch.nn.Conv2d(1, ch1, ks1)
        self.conv2 = torch.nn.Conv2d(ch1, ch2, ks2)
        img_size1 = (img_size - ks1 // 2 * 2) // pool1_size
        img_size2 = (img_size1 - ks2 // 2 * 2) // pool2_size
        ch_in = img_size2 ** 2 * ch2
        if ch_in > 4 or img_size2 <= 0:
            raise Exception('Crazy input')
        self.dense1 = torch.nn.Linear(ch_in, ch3)
        self.dense2 = torch.nn.Linear(ch3, 2)
        self.pool1 = torch.nn.AvgPool2d(pool1_size)
        self.pool2 = torch.nn.AvgPool2d(pool2_size)
```

Here I'm ignoring the cases where the number of inputs is larger than 4, because it will be harder to visualize.

The forward() calls the layers one by one, adding poolings and nonlinearities between them. Here is how the full code for the model looks like:

```python
import torch

class LargeWin(torch.nn.Module):
    def __init__(self, params):
        img_size = 30
        ch1, ch2 = params['ch1'], params['ch2']
        ch3 = params['ch3']
        pool1_size = params['pool1_size']
        pool2_size = params['pool2_size']
        ks1, ks2 = params['ks1'], params['ks2']
        super(LargeWin, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, ch1, ks1)
        self.conv2 = torch.nn.Conv2d(ch1, ch2, ks2)
        img_size1 = (img_size - ks1 // 2 * 2) // pool1_size
        img_size2 = (img_size1 - ks2 // 2 * 2) // pool2_size
        ch_in = img_size2 ** 2 * ch2
        if ch_in > 4 or img_size2 <= 0:
            raise Exception('Crazy input')
        self.dense1 = torch.nn.Linear(ch_in, ch3)
        self.dense2 = torch.nn.Linear(ch3, 2)
        self.pool1 = torch.nn.AvgPool2d(pool1_size)
        self.pool2 = torch.nn.AvgPool2d(pool2_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.tanh(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.tanh(x)

        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)

        return torch.tanh(x)
```

### 2.2 SmallWin

Next we move on to the model with smaller kernels and poolings. It will look very similar to the previous one, except there will be three convolutional layers instead of two, and the size of poolings will be uniform.

```python
class SmallWin(torch.nn.Module):
    def __init__(self, params):
        ch1, ch2 = params['ch1'], params['ch2']
        ch3, ch4 = params['ch3'], params['ch4']
        ks1, ks2 = params['ks1'], params['ks2']
        img_size = 30
        pool_size = 2
        super(SmallWin, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, ch1, ks1)
        self.conv2 = torch.nn.Conv2d(ch1, ch2, ks2)
        self.conv3 = torch.nn.Conv2d(ch2, ch3, ks3)

        img_size1 = (img_size - ks1 // 2 * 2) // pool_size
        img_size2 = (img_size1 - ks2 // 2 * 2) // pool_size
        img_size3 = (img_size2 - ks3 // 2 * 2) // pool_size
        ch_in = img_size3 ** 2 * ch3
        if ch_in > 4 or img_size3 <= 0:
            raise Exception('Crazy input')
        self.dense1 = torch.nn.Linear(ch_in, ch4)
        self.dense2 = torch.nn.Linear(ch4, 2)
        self.pool = torch.nn.AvgPool2d(pool_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.tanh(x)

        x = self.conv2(x)
        x = self.pool(x)
        x = torch.tanh(x)

        x = self.conv3(x)
        x = self.pool(x)
        x = torch.tanh(x)

        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)

        return torch.tanh(x)
```

Note that we check if image size after convolutions is negative. This may happen if we test the network with both large pooling and large kernel. 

### 2.3 Testing the models

Okay, this was a lot of code and we did not run it yet. Let's fix this now. We'll create a main function, that will run a sample input through the model. But first we need to generate this sample, so let's import our generate_dataset.py:

```python
from generate_dataset import generate_image
```

And now add the function itself:

```python
if __name__ == '__main__':
    img, label = generate_image()
    print(img.shape)
```

Next we'll create the model objects. Note that they receive parameters as a dict, so it will look somewhat weird. Let's start with LargeWin model:

```python
if __name__ == '__main__':
    img, label = generate_image()
    print(img.shape)
    modelLargeWin = LargeWin({
        "ch1": 4, 'ch2': 4, 'ch3': 4,
        'pool1_size': 4, 'pool2_size': 2,
        'ks1': 7, 'ks2': 5
    })
```

Next we'll run the image through the model, but first we need to convert it to a tensor, and make it right shape (add examples and channels dimensions):

```python
if __name__ == '__main__':
    img, label = generate_image()
    print(img.shape)
    modelLargeWin = LargeWin({
        "ch1": 4, 'ch2': 4, 'ch3': 4,
        'pool1_size': 4, 'pool2_size': 2,
        'ks1': 7, 'ks2': 5
    })

    img_t = torch.tensor(img).float()
    img_t = img_t.unsqueeze(0).unsqueeze(0)
    outLargeWin = modelLargeWin(img_t)
    print('LargeWin', outLargeWin)
```

This should output original image size and the model output:

```
(30, 30)
LargeWin tensor([[-0.5127,  0.0033]], grad_fn=<TanhBackward>)
```

Now we need to add a similar test for the second model. Here is how the final main function looks:

```python
if __name__ == '__main__':
    img, label = generate_image()
    print(img.shape)
    modelLargeWin = LargeWin({
        "ch1": 4, 'ch2': 4, 'ch3': 4,
        'pool1_size': 4, 'pool2_size': 2,
        'ks1': 7, 'ks2': 5
    })
    modelSmallWin = SmallWin({
        "ch1": 4, 'ch2': 4, 'ch3': 4, 'ch4': 4,
        'ks1': 3, 'ks2': 3, 'ks3': 5
    })

    img_t = torch.tensor(img).float()
    img_t = img_t.unsqueeze(0).unsqueeze(0)
    outLargeWin = modelLargeWin(img_t)
    print('LargeWin', outLargeWin)
    outSmallWin = modelSmallWin(img_t)
    print('SmallWin', outSmallWin)
```

If both models output something healthy (two numbers for sin and cos) - we are ready to move on to training.

## 3. Training

Now having models, and having the data, we are ready to train the models on the data. Let's start by loading the data, converting it to tensors, and creating a dataloader.

```python
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':
    device = torch.device('cuda:0'
        if torch.cuda.is_available()
        else 'cpu')
    data = torch.tensor(np.load('images.npy'))
    labels = torch.tensor(np.load('labels.npy'))
    train_data = data.float().unsqueeze(1).to(device)
    train_labels = labels.float().to(device)
    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, 1024)
    print(len(train_dataloader))
```

Now let's add a function that does actual training. It will accept the model to train and the data to train on. As a result, it will save the trained parameters in a file.

```python
def train_model(model, parameters, dataloader):
    opt = torch.optim.Adam(model.parameters())
    mse = torch.nn.MSELoss()
    for epoch in range(51):
        train_loss = 0
        for input, label in dataloader:
            opt.zero_grad()
            out = model(input)
            loss = mse(out, label)
            train_loss += loss
            loss.backward()
            opt.step()
        train_loss = train_loss / len(dataloader)
        if epoch % 10 == 0:
            print(epoch, train_loss.item())
    params = list(parameters.values())
    params_string = '_'.join([str(i) for i in params])
    file = open('models/{}_{}.pt'.format(
        type(model).__name__, 
        params_string
    ), 'wb')
    torch.save(model.state_dict(), file)
```

Well, it will have to take a 'parameters' argument, to know how to call the model. To test the code, I will create a model object in the main function and train it using train_model(). To do that, I will also need to import the file with the models (assuming it's called 'models.py'). Note that the training script saves the model into a 'models' folder, so we need to create that.

```python
import models
```

```python
    os.makedirs('models', exist_ok=True)
    parameters = {
        "ch1": 2, 'ch2': 2, 'ch3': 2,
        'pool1_size': 3, 'pool2_size': 2,
        'ks1': 5, 'ks2': 7
    }
    model = models.LargeWin(parameters).to(device)
    train_model(model, parameters, train_dataloader)
```

Also note small number of epochs in the training script. By the way, I'm testing the script with a small dataset, so that it doesn't take long to load the data and to train.

After running the script, we have a 'models' folder created and file 'LargeWin_2_2_2_3_2_5_7.pt' inside, which means the training works fine. Let's now log the process, so that we can easily see which models learn better. We will use mlflow for that:

```python
import mlflow
```

I will be using the library in the training script. Will be logging model name and parameters, as well as its loss for each epoch. Here is how the train_model() looks like after we do that:

```python
def train_model(model, parameters, dataloader):
    opt = torch.optim.Adam(model.parameters())
    mse = torch.nn.MSELoss()
    with mlflow.start_run(run_name="OpenCircle"):
        mlflow.log_param('model_type', type(model).__name__)
        for key in parameters:
            mlflow.log_param(key, parameters[key])
        for epoch in range(51):
            train_loss = 0
            for input, label in dataloader:
                opt.zero_grad()
                out = model(input)
                loss = mse(out, label)
                train_loss += loss
                loss.backward()
                opt.step()
            train_loss = train_loss / len(dataloader)
            if epoch % 10 == 0:
                print(epoch, train_loss.item())
                mlflow.log_metric(
                    'train_loss',
                    train_loss.item(), 
                    epoch
                )
        params = list(parameters.values())
        params_string = '_'.join([str(i) for i in params])
        file = open('models/{}_{}.pt'.format(
            type(model).__name__,
            params_string
        ), 'wb')
        torch.save(model.state_dict(), file)
```

Now if we run the code, we should have a new folder called 'mlruns'. If we call a bash command in our folder

```bash
mlflow ui --port=5000
```

and open our browser at localhost:5000 - we should see a report about training one model.

Next we'll need to add a function for trying different combinations for the model parameters and training the models. I other words, we should change this

```python
    parameters = {
        "ch1": 2, 'ch2': 2, 'ch3': 2,
        'pool1_size': 3, 'pool2_size': 2,
        'ks1': 5, 'ks2': 7
    }
    model = models.LargeWin(parameters).to(device)
    train_model(model, parameters, train_dataloader)
```

into this:

```python
    for ch1 in [2, 4]:
        for ch2 in [2, 4]:
            for ch3 in [2, 4]:
                for pool1_size in [2, 3, 4, 8]:
                    for pool2_size in [2, 3, 4, 8]:
                        for ks1 in [5, 7]:
                            for ks2 in [5, 7]:
    # This code should run in the innermost loop
    # I moved it all the way to the right for illustration
    parameters = {
        "ch1": ch1,
        'ch2': ch2,
        'ch3': ch3,
        'pool1_size': pool1_size,
        'pool2_size': pool2_size,
        'ks1': ks1,
        'ks2': ks2
    }
    model = models.LargeWin(parameters).to(device)
    train_model(model, parameters, train_dataloader)
```

Also remember that our model can produce 'Crazy input' exceptions, so we need to try and catch those. At this point I'm going to split the code above into functions, so that it fits in size:

```python
    def run_train_model_large(ch1, ch2, ch3, 
                        pool1_size, pool2_size, 
                        ks1, ks2):
        try:
            parameters = {
                "ch1": ch1,
                'ch2': ch2,
                'ch3': ch3,
                'pool1_size': pool1_size,
                'pool2_size': pool2_size,
                'ks1': ks1,
                'ks2': ks2
            }
            model = models.LargeWin(parameters).to(device)
            train_model(model, parameters, train_dataloader)
        except Exception as e:
             print(e)

    for ch1 in [2, 4]:
        for ch2 in [2, 4]:
            for ch3 in [2, 4]:
                for pool1_size in [2, 3, 4, 8]:
                    for pool2_size in [2, 3, 4, 8]:
                        for ks1 in [5, 7]:
                            for ks2 in [5, 7]:
                                run_train_model_large(
                                    ch1, ch2, ch3,
                                    pool1_size,
                                    pool2_size,
                                    ks1, ks2
                                )
```

If we run the script, after a while we should have a bunch of models exported, and mlflow tracker should indicate this activity. This means that all is going fine. We'll need to add similar script for the second model and we're ready to train real models. The code for the second model has additional ch4 and ks3 parameters, and doesn't have pooling sizes. Except these the code is identical:

```python
    def run_train_model_small(ch1, ch2, ch3, ch4,
                        ks1, ks2, ks3):
        try:
            parameters = {
                "ch1": ch1,
                'ch2': ch2,
                'ch3': ch3,
                'ch4': ch4,
                'ks1': ks1,
                'ks2': ks2,
                'ks3': ks3
            }
            model = models.SmallWin(parameters).to(device)
            train_model(model, parameters, train_dataloader)
        except Exception as e:
             print(e)

    for ch1 in [2, 4]:
        for ch2 in [2, 4]:
            for ch3 in [2, 4]:
                for ch4 in [2, 4]:
                    for ks1 in [3, 5]:
                        for ks2 in [3, 5]:
                            for ks3 in [3, 5]:
                                run_train_model_small(
                                    ch1, ch2, ch3, ch4,
                                    ks1, ks2, ks3
                                )
```

Here is the full code:

```python
import os
import torch
import mlflow
import models
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def train_model(model, parameters, dataloader):
    opt = torch.optim.Adam(model.parameters())
    mse = torch.nn.MSELoss()
    with mlflow.start_run(run_name="OpenCircle"):
        mlflow.log_param('model_type', type(model).__name__)
        for key in parameters:
            mlflow.log_param(key, parameters[key])
        for epoch in range(51):
            train_loss = 0
            for input, label in dataloader:
                opt.zero_grad()
                out = model(input)
                loss = mse(out, label)
                train_loss += loss
                loss.backward()
                opt.step()
            train_loss = train_loss / len(dataloader)
            if epoch % 10 == 0:
                print(epoch, train_loss.item())
                mlflow.log_metric(
                    'train_loss',
                    train_loss.item(),
                    epoch
                )
        params = list(parameters.values())
        params_string = '_'.join([str(i) for i in params])
        file = open('models/{}_{}.pt'.format(
            type(model).__name__,
            params_string
        ), 'wb')
        torch.save(model.state_dict(), file)

if __name__ == '__main__':
    device = torch.device('cuda:0'
        if torch.cuda.is_available()
        else 'cpu')
    data = torch.tensor(np.load('images.npy'))
    labels = torch.tensor(np.load('labels.npy'))
    train_data = data.float().unsqueeze(1).to(device)
    train_labels = labels.float().to(device)
    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, 1024)
    print(len(train_dataloader))

    os.makedirs('models', exist_ok=True)

    def run_train_model_large(ch1, ch2, ch3,
                        pool1_size, pool2_size,
                        ks1, ks2):
        try:
            parameters = {
                "ch1": ch1,
                'ch2': ch2,
                'ch3': ch3,
                'pool1_size': pool1_size,
                'pool2_size': pool2_size,
                'ks1': ks1,
                'ks2': ks2
            }
            model = models.LargeWin(parameters).to(device)
            train_model(model, parameters, train_dataloader)
        except Exception as e:
             print(e)

    for ch1 in [2, 4]:
        for ch2 in [2, 4]:
            for ch3 in [2, 4]:
                for pool1_size in [2, 3, 4, 8]:
                    for pool2_size in [2, 3, 4, 8]:
                        for ks1 in [5, 7]:
                            for ks2 in [5, 7]:
                                run_train_model_large(
                                    ch1, ch2, ch3,
                                    pool1_size,
                                    pool2_size,
                                    ks1, ks2
                                )

    def run_train_model_small(ch1, ch2, ch3, ch4,
                        ks1, ks2, ks3):
        try:
            parameters = {
                "ch1": ch1,
                'ch2': ch2,
                'ch3': ch3,
                'ch4': ch4,
                'ks1': ks1,
                'ks2': ks2,
                'ks3': ks3
            }
            model = models.SmallWin(parameters).to(device)
            train_model(model, parameters, train_dataloader)
        except Exception as e:
             print(e)

    for ch1 in [2, 4]:
        for ch2 in [2, 4]:
            for ch3 in [2, 4]:
                for ch4 in [2, 4]:
                    for ks1 in [3, 5]:
                        for ks2 in [3, 5]:
                            for ks3 in [3, 5]:
                                run_train_model_small(
                                    ch1, ch2, ch3, ch4,
                                    ks1, ks2, ks3
                                )
```

Run it for a small dataset with small number of epochs and make sure that the models are exported, the training is tracked by mlflow, and the script does not crash. If it's ok, generate a larger dataset (I use 256k examples) and run each model for larger number of epochs (I use 5000). It may take some time, the training on my machine takes a couple of weeks. After this is done, we'll be back to analyze the results.

## 4. Analysis

### 4.1 Evaluation

After some models has been trained, we may want to see how they perform. So we will load the same (or maybe newly generated) dataset, and pass them one by one through a selected model. So the main function will look roughly the same as the one in the training script, but with minor changes:

```python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import models

if __name__ == '__main__':
    device = torch.device('cpu')
    data = np.load('images.npy')
    labels = np.load('labels.npy')
    data = torch.tensor(data)\
        .float().unsqueeze(1).to(device)
    labels = torch.tensor(labels).float().to(device)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, 1,  shuffle=True)

    for f in os.scandir('models'):
        print(f)
        model = load_from_file(f)
        eval_model(model, dataloader)
```

So basically we create a dataloader, then look at what models we have. Then one by one, we load these models and pass our dataloader through them. Now we need to implement the missing functions. We start with the one which will load a model from a file. It will look at the file name, and decide which class it is going to instantiate, as well as what parameters it will pass.

```python
def load_from_file(f):
    name_parts = f.name.split('.')[0].split('_')
    model_class_name = name_parts[0]
    if model_class_name == 'LargeWin':
        model_class = models.LargeWin
    elif model_class_name == 'SmallWin':
        model_class = models.SmallWin
    if model_class_name == 'LargeWin':
        parameters = {
            "ch1": int(name_parts[1]),
            'ch2': int(name_parts[2]),
            'ch3': int(name_parts[3]),
            'pool1_size': int(name_parts[4]),
            'pool2_size': int(name_parts[5]),
            'ks1': int(name_parts[6]),
            'ks2': int(name_parts[7])
        }
    else:
        parameters = {
            "ch1": int(name_parts[1]),
            'ch2': int(name_parts[2]),
            'ch3': int(name_parts[3]),
            'ch4': int(name_parts[4]),
            'ks1': int(name_parts[5]),
            'ks2': int(name_parts[6]),
            'ks3': int(name_parts[7])
        }
    model = model_class(parameters)
    model.load_state_dict(torch.load(
        open(f.path, 'rb'),
        map_location='cpu'
    ))

    return model
```

The second function will iterate through the dataloader, pass each image through the model, then plot the image itself along with the model output:

```python
def eval_model(model, dataloader):
    for input, label in dataloader:
        out = model(input)

        # Remove example and channels dimension
        img = input.squeeze(0).squeeze(0).detach().numpy()
        out = out[0].detach().numpy()
        pred_sin = out[0]
        pred_cos = out[1]

        # Plot the input
        plt.subplot(2, 1, 1)
        plt.pcolor(img, cmap='Wistia')

        # Plot the output
        plt.subplot(2, 1, 2)
        plt.plot([-1, 1], [pred_sin, pred_sin])
        plt.plot([pred_cos, pred_cos], [-1, 1])
        plt.gca().set_xticks([-1, 0, 1])
        plt.gca().set_yticks([-1, 0, 1])
        plt.grid()
        plt.gcf().set_size_inches(3, 6)
        plt.show()
```

Note that we are plotting the output as lines, corresponding to sin and cos prediction. By looking at the image we can decide if the model performs good enough:

![01_eval](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\01_eval.png)

By doing this we can visually understand what does the final loss mean. If we have RMSE 0.12 after training, is it good or bad. Or to judge if the model predicts wrong really noisy images, or is bad in general. Ultimately, to see if it work at all. 

### 4.2 Performance statistics

Now that we have trained a bunch of models, we can see which parameters improve the model performance. To do that, we will find correlation scores between the loss at the end of the training and the model parameters (filters number and pooling size).

To start with, let's download the mlflow logs. In the mlflow web app, click "download csv" and save in the project directory:

![runs](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\runs.png)

Now we need to load the file in python with pandas:

```python
import pandas as pd

data = pd.read_csv("runs.csv")
data = data[[
    'model_type',
    'Status',
    'ch1', 'ch2', 'ch3',
    'ks1', 'ks2',
    'pool1_size', 'pool2_size'
]]
print(data.shape)
```

Note that we do not need all the columns, so we list the ones that will be useful. Now let's filter only one model to analyze (LargeWin, to start with) and the models that have not failed (whose status is 'finished'):

```python
data = data[data['Status'] == 'FINISHED']
data = data[data['model_type'] == 'LargeWin']
print(data.shape)
```

And finally, calculate the correlations:

```python
print(data.corr(method='pearson')['train_loss'])
```

Which gives the following output (for me)

```
train_loss    1.000000
ch1          -0.390550
ch2          -0.252477
ch3          -0.031024
ks1          -0.430135
ks2           0.114280
pool1_size   -0.407319
pool2_size    0.133425
```

This tells us that the larger pool1_size, for example, we use - the smaller will be the loss. But if we take large pool2_size - the loss tend to be larger. Let's make a couple of plots:

```python
import matplotlib.pyplot as plt

plt.scatter(data['pool1_size'], data['train_loss'], 
            label="Pool1", alpha=0.2)
plt.scatter(data['ks2'], data['train_loss'], 
            label="Ks2", alpha=0.2)
plt.legend()
plt.show()
```

This will give an image like this:

![corr](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\corr.png)

We see that the model on average performs better with smaller ks2, but the best result was obtained with a large ks2, which is interesting.

After this result is obtained, I will select the models I want to keep. I will modify the training script, the part in which we test different parameter sets:

```python
    for ch1 in [4]:
        for ch2 in [2, 4]:
            for ch3 in [2]:
                for pool1_size in [3, 4]:
                    for pool2_size in [2]:
                        for ks1 in [7]:
                            for ks2 in [5, 7]:
                                run_train_model_large(
                                    ch1, ch2, ch3,
                                    pool1_size,
                                    pool2_size,
                                    ks1, ks2
                                )
```

The same procedure will be repeated for the second model, and we'll move on to analyzing the trained weights.

### 4.3 Convolution filters

Now that we know that our models are working, let's find out how do they work. We will start with the initial layers, which are convolutions. To analyze their work we will write a script that shows every filter and the model output after each convolution layer. Apart from a plotting tool, we will need to make changes to the models script. First, we have to enable intermediate output, if we want to see it. Second, our load_from_file() function we used earlier for evaluation, will be needed here as well, so I will move it to the models file.

To enable the model's intermediate output, we will add a new parameter to its forward() function. Then inside the function, we check the parameter for early output:

```python
    def forward(self, x, out_layer=-1):
        x = self.conv1(x)
        if out_layer == 0:
            return x
        x = self.pool1(x)
        x = torch.tanh(x)

        x = self.conv2(x)
        if out_layer == 1:
            return x
        x = self.pool2(x)
        x = torch.tanh(x)

        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)

        return torch.tanh(x)
```

I repeat the procedure for the SmallWin model. The operation is same, except it has 3 convolution layers, so it has three checks. 

Now we may return to the plotting script and sketch its skeleton:

```python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_dataset import generate_image
import models

np.random.seed(42)

img, lbl = generate_image()
t_img = torch.tensor(img).float() \
    .unsqueeze(0).unsqueeze(0)
for f in os.scandir('models'):
    print(f)
    model = models.load_from_file(f)
    out = model(t_img).detach().reshape(-1).numpy()

    print(out)
    print(lbl)
    plt.pcolor(img, cmap='Wistia')
    plt.show()

    show_weight(model.conv1.weight, model(t_img, 0)[0])
    show_weight(model.conv2.weight, model(t_img, 1)[0])
```

Here we are generating an image (setting a seed for random, so that the image is same), loading a model, passing the image through the model, and plotting the information. The source image for all the illustrations below looks like this:

![src](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\src.png)

The show_weight() function, which we will write shortly, accepts as arguments the weight to plot and the model output after the weight is applied. It contains some plotting commands along with some logic to arrange the images:

```python
def show_weight(weight, output):
    plot_cnt = 1
    img_width = weight.shape[0]
    img_height = weight.shape[1]
    img_height += 1  # The outputs row

    # Plot weights
    for i in range(weight.shape[1]):
        for j in range(weight.shape[0]):
            filter = weight[j, i].detach().numpy()
            plt.subplot(img_height, img_width, plot_cnt)
            plt.pcolor(filter, cmap='Wistia')
            plot_cnt += 1

    # Plot outputs - at the bottom
    for j in range(weight.shape[0]):
        plt.subplot(img_height, img_width, plot_cnt)
        out_img = output[j].detach().numpy()
        plt.pcolor(out_img, cmap='Wistia')
        plot_cnt += 1

    plt.tight_layout(0.1, 0.1, 0.1)
    plt.gcf().set_size_inches(img_width * 2, img_height * 2)
    plt.show()
```

If we run the script, we'll be getting images like these:

![filters_1](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_1.png)

![filters_2](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_2.png)

These are the first two layer of one model. The last row shows the output of the layer, while the first rows show the filters applied to the previous layer output. For the first layer we see some kind of pattern for filter, and some sort of processing for the image. But for the second filter, both filter weights and output look like complete noise, which is a sign of overfitting. To deal with this, we may use some regularization technique, like L2 regularization. In PyTorch, we need to add a weight_decay parameter for optimizer:

```python
opt = torch.optim.Adam(model.parameters())
# Change to:
opt = torch.optim.Adam(
    model.parameters(), 
    weight_decay=1e-2
)
```

After that, we need to train our models again. Luckily, we have found which parameters correlate most with the loss, so this training will take much less time. I have created other training script, similar to the first one, but included all the changes so far:

```python
import os
import torch
import mlflow
import models
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def train_model(model, parameters, dataloader):
    opt = torch.optim.Adam(
        model.parameters(),
        weight_decay=1e-2
    )
    mse = torch.nn.MSELoss()
    with mlflow.start_run(run_name="OpenCircleWD"):
        mlflow.log_param('model_type', type(model).__name__)
        for key in parameters:
            mlflow.log_param(key, parameters[key])
        for epoch in range(5001):
            train_loss = 0
            for input, label in dataloader:
                opt.zero_grad()
                out = model(input)
                loss = mse(out, label)
                train_loss += loss
                loss.backward()
                opt.step()
            train_loss = train_loss / len(dataloader)
            if epoch % 10 == 0:
                print(epoch, train_loss.item())
                mlflow.log_metric(
                    'train_loss',
                    train_loss.item(),
                    epoch
                )
        params = list(parameters.values())
        params_string = '_'.join([str(i) for i in params])
        file = open('models/{}_{}_wd.pt'.format(
            type(model).__name__,
            params_string
        ), 'wb')
        torch.save(model.state_dict(), file)

if __name__ == '__main__':
    device = torch.device('cuda:0'
        if torch.cuda.is_available()
        else 'cpu')
    data = torch.tensor(np.load('images.npy'))
    labels = torch.tensor(np.load('labels.npy'))
    train_data = data.float().unsqueeze(1).to(device)
    train_labels = labels.float().to(device)
    train_dataset = TensorDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, 1024)
    print(len(train_dataloader))

    os.makedirs('models', exist_ok=True)

    def run_train_model_large(ch1, ch2, ch3,
                        pool1_size, pool2_size,
                        ks1, ks2):
        try:
            parameters = {
                "ch1": ch1,
                'ch2': ch2,
                'ch3': ch3,
                'pool1_size': pool1_size,
                'pool2_size': pool2_size,
                'ks1': ks1,
                'ks2': ks2
            }
            model = models.LargeWin(parameters).to(device)
            train_model(model, parameters, train_dataloader)
        except Exception as e:
             print(e)

    for ch1 in [4]:
        for ch2 in [2, 4]:
            for ch3 in [2]:
                for pool1_size in [3, 4]:
                    for pool2_size in [2]:
                        for ks1 in [7]:
                            for ks2 in [5, 7]:
                                run_train_model_large(
                                    ch1, ch2, ch3,
                                    pool1_size,
                                    pool2_size,
                                    ks1, ks2
                                )

    def run_train_model_small(ch1, ch2, ch3, ch4,
                        ks1, ks2, ks3):
        try:
            parameters = {
                "ch1": ch1,
                'ch2': ch2,
                'ch3': ch3,
                'ch4': ch4,
                'ks1': ks1,
                'ks2': ks2,
                'ks3': ks3
            }
            model = models.SmallWin(parameters).to(device)
            train_model(model, parameters, train_dataloader)
        except Exception as e:
             print(e)

    for ch1 in [4]:
        for ch2 in [4]:
            for ch3 in [2, 4]:
                for ch4 in [2, 4]:
                    for ks1 in [3]:
                        for ks2 in [3, 5]:
                            for ks3 in [5]:
                                run_train_model_small(
                                    ch1, ch2, ch3, ch4,
                                    ks1, ks2, ks3
                                )
```

After a while, we get another set of trained models. For me, if I visualize the weights, it looks something like this:

![filters_21](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_21.png)

![filters_31](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_31.png)

These are the samples from the first layer of different models. Here is what the second layer look like:

![filters_22](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_22.png)

![filters_32](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_32.png)

This looks much better. All the filters are symmetric, and the pattern (distribution of lighter and darker parts) is clear. We can move on to figuring out how it actually works.

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