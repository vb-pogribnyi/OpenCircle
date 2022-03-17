# CNN Intermediate - OpenCircle

Bigger than linear regression, but simpler than MNIST

## Introduction

We all know that convolutional neural networks are superior for processing images. So given a dataset of images like the one below, they could be easily trained to tell at which point the circle is broken.

![dataset_sample](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\dataset_sample.png)

But how does it actually do it? Theoretically, I may generate a synthetic dataset, train the network, then apply it to real data like this (I've drawn these myself with a pencil):

![real_sample](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\real_sample.png)

But many of us know that the network trained on a synthetic dataset doesn't necessarily work for the real data. 

Trying to answer this question, I made a little research, about which I'm telling in this article. I trained a network on a synthetic dataset, analyzed its performance, made it work for real data.

## 1. Dataset

The dataset consists of images of open circles 30x30 pixels. There will be something I call 'core image', which is a collection of x-y coordinates of points, meant to be connected by lines. This core image will be then drawn on a bitmap, a noise will be added to it, and it will be fed to the NN model.

![dataset_pipeline](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\dataset_pipeline.png)

On the picture above: core image, bitmap, noisy bitmap.

The core image will represent a circle itself. But since I want to generate all kinds of circle variations, including ovals, circles with linear edges (half circle - half rectangle) - I will need to introduce some additional logic. After the image is generated - it will be scaled and rotated at a random angle, because, again, we want to generate all kinds of circles.

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

The get_corner_points() function builds the curve itself. It accepts the number of points generated, as parameter. The function is called this way because it generates 'corners' of our circle. If I mirror them around the image, I will get a complete circle. In other words, if I changed the main code this way:

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

The image on the right shows what we have now. Notice that if we follow the order of the points counter-clockwise, we start at corner 4, first point, then meet corner 4 last point, then corner 3 last point, corner 3 first point, corner 1 first point, and so on. We need to reorder things around so they look like on the image on the left:

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

Looks fine. Now let's collect all the points in one array, by concatenation. There is only one issue. I want the array's first point to be the one at 'zero degrees', or the one marked with a red cross below:

![circle_7](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\01_dataset\circle_7.png)

This will make me divide line41 into halves, then concatenate the points:

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

Now we need to make sure that the distance between the points is uniform. The easy way to do it would be to use numpy interp function. To do that, I will create an array of distances between points for the current circle, then an array of desired (uniform) distances, and make the interpolation:

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

Notice that image on the left is cut by 30% (since we have this code):

```python
plt.plot(xs[15:-15], ys[15:-15])
```

The gap has become larger, because we have smaller amount of points overall, but still removing 15 of them.

Now I will add to this function only a return value. The final code for the function looks like this:

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

To generate the 'all kinds of open circles, let's take what we have, and scale it at a random value. And also rotate. The operation will be done by matrix multiplication. The function responsible for this transform will be accepting the following parameters:

- x and y coordinates of our circle
- rotation angle
- scale x, scale y

So this would be a stub for the function:

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

The function we're about to write will be accepting an array of point coordinates (which we already have), and output our final 30x30 image. We will use OpenCV for that, so the first thing is, obviously, to import it:

```python
import cv2 as cv
```

Next let's a stub for our function:

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

Now let's add drawing procedure. Keep in mind that our core image is in range (0, 1) - this should be mapped to (0, 30), otherwise our image will appear as a single dot:

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

These types of noise are applied before the image is scaled. Third type of noise will be a white noise, and will be applied to the whole image after it is drawn:

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

Now that we're able to create an image given some parameters, let's create a function that will generate those parameters and consequently generate the image. This function will also give actual 'opennes' to the image, and generate label for it. The function will accept no parameters, since it is going generate the parameters, and will output the image along with its label. The labels will be given in form of sin and cos of the angle. We don't want to use the angle itself because it will be hard for the network to figure out the importance of the error. 

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

Now we have to implement the randomizations. Let's start with the angle, so that we can implement the label as well. We need to generate the angle in degrees, then transform it into radians for the label. We need to insert it into the transform function as well:

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

This gives us a circle that is open in 20% of its length. Note that if we change the number of points for the core image:

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

### 2.1. Architecture

For our purposes, we will use a 2-layer CNN network, followed by a dense layer. We will have relatively large kernels (5-7 px) and large poolings (2-4 px). So the whole architecture may be described like this:

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

I'm passing the parameters here as a dict, so that it will be easier to automate testing later on. Among the parameters, there are ch1, ch2 - number of filters for the convolutions. Ch3 - number of hidden neurons for the dense network. Pool1_size and pool2_size - sizes of the pooling layers after the convolutions. Ks1 and ks2 - convolutions kernel sizes. 

Let's now add the network layers. Note that dense layer input size (after flattening) will depend on the size of kernels and pooling windows. It will be calculated in the model constructor as ch_in:

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

The forward() method calls the layers one by one, adding poolings and nonlinearities between them. Here is what the full code for the model looks like:

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

### 2.2 Testing the model

Okay, this was a lot of code and we did not run it yet. Let's fix this now. We'll create the main function, that will run a sample input through the model. But first we need to generate this sample, so let's import our generate_dataset.py:

```python
from generate_dataset import generate_image
```

And now add the function itself:

```python
if __name__ == '__main__':
    img, label = generate_image()
    print(img.shape)
```

Next we'll create the model objects. Note that they receive parameters as a dict, so it will look somewhat weird:

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

Next we'll run the image through the model, but first we need to convert it to a tensor and make it the right shape (add examples and channels dimensions):

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

If the model outputs something healthy (two numbers for sin and cos) - we are ready to move on to training.

## 3. Training

Now having the model, and having the data, we are ready for the training. Let's start by loading the data, converting it to tensors, and creating a dataloader.

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

Now the function for actual training. Nothing fancy so far. Train the model, save the weights in a file.

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

To test this script, I simply create a model and run it through the training function. Note that it will attempt to save the weight in a 'models' folder, so we need to create it.

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

Also note the small number of epochs in the training script. By the way, I'm testing the script with a small dataset, so that it doesn't take long to load the data and to train.

After running the script, we have a 'models' folder created and file 'LargeWin_2_2_2_3_2_5_7.pt' inside, which means the training works fine. Let's now log the process, so that we can easily see which models learn better. We will use mlflow for that:

```python
import mlflow
```

It will be logging model name and parameters, as well as the loss for each epoch. Here is how the train_model() looks like after I do that:

```python
def train_model(model, parameters, dataloader):
    opt = torch.optim.Adam(model.parameters())
    mse = torch.nn.MSELoss()
    with mlflow.start_run(run_name="OpenCircle"):
        # Log the model name
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
                # Log the progress
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

Next we'll need to try different parameters for the model training. I other words, we should change this

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

If we run the script, after a while we should have a bunch of models exported, and mlflow tracker should indicate this activity. This means that all is going fine. 

Run it for a small dataset with small number of epochs and make sure that the models are exported, the training is tracked by mlflow, and the script does not crash. If it's ok, generate a larger dataset (I use 256k examples) and run the model for larger number of epochs (I use 5000). It may take some time, the training on my machine takes a couple of weeks. After this is done, we'll be back to analyze the results.

## 4. Analysis

### 4.1 Evaluation

After some models have been trained, we may want to see how they perform. So we will load the same (or maybe newly generated) dataset, and pass it through a selected model. So the main function will look roughly the same as the one in the training script, but with minor changes:

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

So basically we create a dataloader, then look at what models we have. Then one by one, we load these models and pass our dataloader through them. 

Now we need to implement the missing functions, like the one that loads a model from a file. It will look at the file name, and decide how to create the model object.

```python
def load_from_file(f):
    name_parts = f.name.split('.')[0].split('_')
    parameters = {
        "ch1": int(name_parts[1]),
        'ch2': int(name_parts[2]),
        'ch3': int(name_parts[3]),
        'pool1_size': int(name_parts[4]),
        'pool2_size': int(name_parts[5]),
        'ks1': int(name_parts[6]),
        'ks2': int(name_parts[7])
    }
    model = models.LargeWin(parameters)
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

Note that we are plotting the output as lines, corresponding to sin and cos prediction. So we're looking at the place those lines cross. By looking at the image we can decide if the model performs good enough:

![01_eval](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\01_eval.png)

Also, we visually understand what does the final loss mean. If we have MSE of 0.12 after training - whether it is good or bad. Or to judge if the model predicts wrong really noisy images, or is bad in general. Ultimately, to see if it work at all. 

### 4.2 Performance statistics

Now that we have trained a bunch of models, we can see which parameters affect the model performance. Afterward, we may choose a set of parameters for our final model.

To do that, we will find correlation scores between the loss at the end of the training and the model parameters (filters number and pooling size).

To start with, let's download the mlflow logs. In the mlflow web app, click "download csv" and save in the project directory:

![runs](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\runs.png)

And load the file in python with pandas:

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

This tells us that the larger pool1_size, for example, we use - the smaller will be the loss. But if we take large pool2_size - the loss tends to be larger. Let's make a couple of plots:

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

I plan to keep experimenting with the models, so I will choose the ones with the most promising parameters. I will modify the training script, the part in which we test different parameter sets:

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

I won't run the training again, yet. First we need to take a look at the trained filters, because we may need to update the training process.

### 4.3 Convolution filters

Now that we know that our model is working, let's find out how does it work. We start with the convolutions. And since the convolutions are composed of filters, the easy way to show its operation would be to show the image after each filter is applied. For example, if we have a convolution layer with 4 filters and one channel input image, we want to see 4 images: each of the filters applied to the image. If we have 2 channels input - we will get 8 images: each of the filters applied to each of the layers of the image.

We'll need a plotting tool for that. But apart from that, we will need to make changes to the model's script, to enable intermediate output (output of the first layer, for example). Second, some code organization. Our load_from_file() function we used earlier for evaluation, will be needed here as well, so I will move it to the model's file.

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

Now we may return to the plotting script, and sketch its skeleton:

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

Here we are generating an image (setting seed for np.random, so that the image is the same), loading a model, passing the image through the model, and plotting the information. The source image for all the illustrations below looks like this:

![src](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\src.png)

The show_weight() function accepts as arguments the weight to plot and the model output after the weight is applied. It contains some plotting commands along with some logic to arrange the images:

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

For the first layer (above), and the second layer below:

![filters_2](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_2.png)

The last row shows the output of the layer, while the first rows show the filters applied to the previous layer output. For the first layer, we see some kind of pattern for the filter, and its output looks meaningful. But for the second filter, both filter weights and output look like complete noise, which is a good sign of overfitting. To deal with this, we may use some regularization techniques, like L2 regularization. To do that, in PyTorch, we need to add a weight_decay parameter for optimizer:

```python
opt = torch.optim.Adam(model.parameters())
# Change to:
opt = torch.optim.Adam(
    model.parameters(), 
    weight_decay=1e-2
)
```

After that, we need to train our models again. Luckily, we have found which parameters correlate most with the loss, so this training will take much less time. I have created another training script, similar to the first one, but included all the changes so far:

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
```

After a while, we get another set of trained models. For me, if I visualize the weights, it looks something like this:

![filters_21](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_21.png)

Or for another model:

![filters_31](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_31.png)

Here is what the second layer look like:

![filters_22](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_22.png)

Another model:

![filters_32](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_32.png)

This looks much better. All the filters are symmetric, and the pattern is clear. We can move on to exploring the dense network part.

## 5. Breakdown

This section aims to illustrate how the different parts of the model work.

### 5.1 Dense layers

We will take the simplest network configuration for illustration purposes: 2 inputs, 2 outputs, and 2 hidden nodes. Note that we don't have to account for the hidden nodes. We may only plot the very output vs the input. So there will be 2 plots, one for each of the outputs. The 2 inputs will be denoted as x and y, the output - as color. The input values vary between -1 and 1, since the activation on the previous layer is tanh. Here's what it looks like in code:

```python
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import models


def show_dense(model):
    size = 10
    for channel in range(2):
        plt.subplot(2, 1, channel+1)
        plt.title('Sin' if channel == 0 else 'Cos')
        data = np.zeros((size, size))
        space = np.linspace(-1, 1, size)
        for idx_i, i in enumerate(space):
            for idx_j, j in enumerate(space):
                model_in = torch.tensor([i, j]).float()
                out = model(model_in, in_layer=1)[channel]
                data[idx_j, idx_i] = out.item()
        plt.pcolor(data, cmap='hot', vmin=-2, vmax=2)
        plt.xticks([0, 5, 10], [-1, 0, 1])
        plt.yticks([0, 5, 10], [-1, 0, 1])
    plt.gcf().set_size_inches(5, 10)
    plt.show()

if __name__ == '__main__':
    for f in os.scandir('models'):
        if f.name != 'LargeWin_4_2_2_3_2_7_7_wd.pt':
            continue
        print(f)
        model = models.load_from_file(f)
        show_dense(model)
```

Note that here I'm setting the model weights for which I want to print the data (LargeWin_4_2_2_3_2_7_7_wd). This code will output the following image:

![dense_2](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\dense_2.png)

The red-ish colors here mean values closer to -1, the yellow-ish to +1. Let's take a look at the bottom (cos) image. If the inputs to the network were (-1, 1), top left corner, the output would be -1. If the inputs were (+1, -1) - we would appear in the bottom right corner and output +1. For input like (+1, +1) - the network would output a small value close to 0. The same considerations are applicable for the top plot, of course.

### 5.2 General scheme

The chart below shows briefly the overall operation of the network:

![filter_chain_expanded](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filter_chain_expanded.png)

So we have an input image, it is first passed through a set of 4 filters (Filter 1 on the chart, or conv1 in our network). This gives us 4 filtered images, which are then passed through another set of filters Filter 2. Since our conv2 layer gives 2 output channels, it has 8 filters in total. The chart shows only 4, belonging to the first out channel. The output of these 4 filters gets summed up, because this is how the convolution layer works. After that we get a 2x2 image, 4 pixels total. The average pooling simply takes the average of those pixels, yielding a single number. This number becomes one of the inputs to the dense network.

Now that things are getting a bit complicated, let's take an example and try to find what led the network to such a result.

So, say our network outputs a "90°" prediction for an image. For that, the cos output should be close to "+1", the sin output - to "0". Take a look again at the dense network plots above. We get such a result when the output of the convolution is close to (-1, +1). 

To get the "90°" output, the first set of filters (belonging to output channel 0, and showed on the chart), has to output "-1", the other set - "+1". 

To me, after I've done all of that, the question remained, how the filter would output these 4 pixels, that after averaging would become the decisive "-1" and "+1". For that, I wrote another script, which illustrates the behavior of a filter. Let's take a look at its output first:

![filters_n_1_marks](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_n_1_marks.png)

It illustrates the work of one of the filters in the second convolution layer. We got a text description in the top left corner, it shows which one of the 8 filters we're looking at. Right below it, marked "Original" - is the input image before pooling; and the one to the right ("Src") is the same image after pooling. 

As we know, to apply a filter means to multiply the filter by the parts of the image, marked by different colors on the plot. These parts of the image, multiplied by the filter, are marked "Out 1" to "Out 4". After the multiplication, the values of each "Out" are summed up to form 4 single values in total (or image 2x2). This will be the output for a single filter, but for the convolution layer we have four filters, and their outputs are summed up. This will be a single channel output from the convolution and denoted in the plot above as "Output", in the lower-left corner.

So how is this connected to our example? In order for the first channel to output the "-1" - all the four filters with "Filter out idx: 0" have to output a large negative value (remember we have a tanh non-linearity, so the large negative will be converted to -1). Well, not all of them, but mostly, on average.

And, how would a single filter output a large negative value? For that, we need to take a look at the numbers themselves. Interpretation of the numbers is also straightforward: These are only multiplications of these image parts by the filter values:

![filters_n_1_nums](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_n_1_nums.png)

However, you may want to take a second to figure out which number goes where (use the previous plot for reference).

Let's think about what we see here. First of all, the filter zeroes out most of the values in the middle, leaving mostly top right and bottom left. This means that these top right and bottom left parts of the image actually affect the filter output, obviously. The second is, this filter "likes" when the top right values are negative and the bottom left are positive. In this case, the filter outputs a larger value. If our image had positive values both at the top right and bottom left - they would cancel out, which means that the filter wouldn't care about them (remember that after the filter is applied, its outputs will sum up).

 Let's look at another filter operation:

![filters_num_2](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_num_2.png)

This filter also does not pay much attention to the center pixels, mostly edge pixels affect its output. For our example, when the image is open at 150-ish degrees, its right and bottom edges are darker, compared to the bottom and right, so such filter would output something rather negative.

Now assume that we understand how the "Output" matrix is formed. The average value of this matrix would be the first input to our dense network. If we remind ourselves of how the dense network work, we may connect the filters output to the "Grand output" - output of the whole network. So, if all the filters output a positive value, the network tends to output "Large Sin, zero Cos". If some filters output positive value, others negative, in such a manner that they compensate each other - the network will output zero for both Sin and Cos, which should never happen, theoretically.

Next, I will describe the script for obtaining this visualization, so that you can reproduce them and play with it yourself. But before it, one small point to mention.

### 5.3 More values from inside

We are going to explore our model even further. We want to see an image before convolution, after it, and after pooling. For that, we need to change the second parameter for the forward() function:

```python
    def forward(self, x, out_layer=-1):
        x = self.conv1(x)
        if out_layer == 11:
            return x
        x = self.pool1(x)
        x = torch.tanh(x)
        if out_layer == 12:
            return x

        x = self.conv2(x)
        if out_layer == 21:
            return x
        x = self.pool2(x)
        if out_layer == 22:
            return x
        x = torch.tanh(x)

        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)

        return torch.tanh(x)
```

The default value for the parameter (-1) is something invalid, so that the model works from the beginning to the end, unless specified otherwise.

### 5.4 The illustration code

First of all, the script will draw the matrices, like inputs and outputs for the filter, but it will also write a number ontop of every pixel, so that we can see the actual value. So we start by writing a function for it:

```python
import numpy as np
from generate_dataset import generate_image
import matplotlib.pyplot as plt



def show_matrix(arr, ax, title='', text=None):
    arr = arr[::-1]
    ax.set_title(title)
    ax.matshow(arr, cmap='Wistia')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if text is None:
                ax.text(j, i, str(round(arr[i, j], 2)), 
                        va='center', ha='center')
            else:
                ax.text(j, i, text, 
                        va='center', ha='center')
    ax.set_axis_off()

    return ax

# Setting a seed, so that we generate the same image
np.random.seed(42)
img, lbl = generate_image()

fig, ax = plt.subplots(1, 2)
show_matrix(img, ax[0])
show_matrix(img[5:15, 5:15], ax[1])
plt.show()
```

To test it, we print our input image. But since the image has too many pixels, the text ontop merges and we see nothing. So I plotted a fraction of this image nearby.

The code should produce the following image:

![numerical_input](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\numerical_input.png)

The image is the same input as on the previous plot. The right image is a zoomed-in version of the left one.

Next we add a function that shows actual plots. We will have 9 images overall, organized in a 3x3 grid. Let's start by showing description, input and filter.

```python
def illustrate(model, img, filter_in_idx, filter_out_idx):
    filter = model.conv2 \
        .weight[filter_out_idx, filter_in_idx] \
        .detach().numpy()

    fig, ax = plt.subplots(3, 3)
    description = f"Filter in idx: {filter_in_idx} \n " \
                  f"Filter out idx: {filter_out_idx} \n "
    show_matrix(np.array([[0]]), ax[0, 0], "", description)
    show_matrix(filter, ax[0, 2], "Filter")
    fig.set_size_inches(10, 10)
    plt.tight_layout()
    plt.show()
```

Now we only have to call this function with our selected model, and set filter indexes to display. Here's how I did that:

```python
for filter_out_idx in [0, 1]:
    for filter_in_idx in [0, 1, 2, 3]:
        for f in os.scandir('models'):
            if f.name != 'LargeWin_4_2_2_3_2_7_7_wd.pt':
                continue
            print(f)
            model = models.load_from_file(f)
            illustrate(
                model, img, 
                filter_in_idx, filter_out_idx
            )
```

This code should show a plot with two images, nothing too fancy. Let's add actual images from inside the network. I will add the output image from the previous layer (call it "Original"), the input to the convolution (called "Src"), and the output image ("Output"). Here's the code:

```python
def illustrate(model, img, filter_in_idx, filter_out_idx):
    filter = model.conv2 \
        .weight[filter_out_idx, filter_in_idx] \
        .detach().numpy()
    img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    img_orig = model(
        img_t, 
        out_layer=11
    )[0, filter_in_idx].detach().numpy()
    layer_in = model(
        img_t, 
        out_layer=12
    )[0, filter_in_idx].detach().numpy()
    out_nopool = model(
        img_t, 
        out_layer=21
    )[0, filter_out_idx].detach().numpy()

    fig, ax = plt.subplots(3, 3)
    description = f"Filter in idx: {filter_in_idx} \n " \
                  f"Filter out idx: {filter_out_idx} \n "
    show_matrix(np.array([[0]]), ax[0, 0], "", description)
    show_matrix(filter, ax[0, 2], "Filter")
    show_matrix(img_orig, ax[1, 0], "Original", '')
    show_matrix(out_nopool, ax[2, 0], "Output")
    show_matrix(layer_in, ax[0, 1], "Src")
    fig.set_size_inches(10, 10)
    plt.tight_layout()
    plt.show()
```

Note that I've added an empty string as a text field for the Original image, so that the numbers are not written, this would overwhelm the image.

Now it's time to add the convolution function. It will accept an image and a filter, like regular convolution. It will split the image input pieces sized the same as the filter (7x7 in our case), and multiply by the filter. 

```python
def convolve(arr, filter):
    shifts_x = arr.shape[0] - filter.shape[0] + 1
    shifts_y = arr.shape[1] - filter.shape[1] + 1
    result = []
    for i in range(shifts_x):
        conv_row = []
        for j in range(shifts_y):
            conv_item = arr[
                        i:i + filter.shape[0], 
                        j:j + filter.shape[1]
                        ]
            conv_row.append(conv_item * filter)
        result.append(conv_row)
    return result
```

Since the second layer input is 8x8, the filter will be applied to 4 sub-images, and the function will return an array of 2x2. Note that 'conv_item * filter' will return a matrix where every element of conv_item is multiplied by the respective element of the filter.

When the function is ready, we may plot its values in 'illustrate()':

```python
def illustrate(model, img, filter_in_idx, filter_out_idx):
    filter = model.conv2 \
        .weight[filter_out_idx, filter_in_idx] \
        .detach().numpy()
    img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

    img_orig = model(
        img_t,
        out_layer=11
    )[0, filter_in_idx].detach().numpy()
    layer_in = model(
        img_t,
        out_layer=12
    )[0, filter_in_idx].detach().numpy()
    out_nopool = model(
        img_t,
        out_layer=21
    )[0, filter_out_idx].detach().numpy()
    conv = convolve(layer_in, filter)

    fig, ax = plt.subplots(3, 3)
    description = f"Filter in idx: {filter_in_idx} \n " \
                  f"Filter out idx: {filter_out_idx} \n "
    show_matrix(np.array([[0]]), ax[0, 0], "", description)
    show_matrix(filter, ax[0, 2], "Filter")
    show_matrix(img_orig, ax[1, 0], "Original", '')
    show_matrix(out_nopool, ax[2, 0], "Output")
    show_matrix(layer_in, ax[0, 1], "Src")
    show_matrix(conv[1][0], ax[1, 1], "Out 1")
    show_matrix(conv[1][1], ax[1, 2], "Out 2")
    show_matrix(conv[0][0], ax[2, 1], "Out 3")
    show_matrix(conv[0][1], ax[2, 2], "Out 4")
    fig.set_size_inches(10, 10)
    plt.tight_layout()
    plt.show()
```

The code will output something like this:

![filters_num_1](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\04_analysis\filters_n_1.png)

Ugh, this was a tough section. But I believe this is a must-have, because it would lead us to understanding of the process, which would allow us to apply it to the real data. We will do it in the next section.

##  5. Real data

Any ML model is useless without application to a real application (or real data). The usefulness of this particular model is still questionable, but it's still nice to know that it works for real data.

### 5.1 Evaluation

To collect the data for testing, I draw a couple of open circles on paper and took a photo of them:

![img_real](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\img_real.jpg)

Then cut it in a drawing software into pieces that remind our dataset:

![02](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\01.png)

Now there are a couple of problems that prevent us from feeding this directly into the model. First, this image is too large and has to be scaled down to 30x30. Second, this image is not square, so we need to crop it first.

Let's write a script that does it for us:

```python
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for f in os.scandir('data_real'):
        img = cv.imread(f.path, cv.IMREAD_GRAYSCALE)
        img_size = np.min(img.shape)
        img = img[:img_size, :img_size]
        img = img.astype(float)
        img = cv.resize(img, (30, 30))
        img = img - np.min(img)
        img = img / np.max(img)
        img = 1 - img

        plt.pcolor(img, cmap='Wistia')
        plt.show()
```

The code suggests that all the images we cut by hand should be in a folder called 'data_real'. OpenCV reads images as an array of integers with values in the range of 0-255. Since our network accepts arrays with values 0-1 - we also have to convert the images to floats and normalize them, by subtracting min value from each image and dividing it by max value. Also since on our images, the circle is black-on-white, and in the dataset the circle is white-on-black, we end our preprocessing with the line 'img = 1 - img', which inverts the image.

Here's an example of what the code should output:

![03](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\im_02.png)

There is some inconsistency between OpenCV and Matplotlib for the y-axis, so the image looks flipped vertically. This is the same image as in the plot above.

Now given the data, we can transform it into a PyTorch-friendly dataloader and run it through our model. To run it through the model, we'll use our function from the eval algorithm, which accepts a model and a dataloader. This seems ok, except this function accepts a dataset that also includes labels, which we don't have for our real data. Well that's not a problem, we can substitute the real labels with empty (zeros) tensor. Here's what the updated code looks like:

```python
import os
import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import models
from eval import eval_model

if __name__ == '__main__':
    device = torch.device('cpu')
    data = []
    for f in os.scandir('data_real'):
        print(f.name)
        img = cv.imread(f.path, cv.IMREAD_GRAYSCALE)
        img_size = np.min(img.shape)
        img = img[:img_size, :img_size]
        img = img.astype(float)
        img = cv.resize(img, (30, 30))
        img = img - np.min(img)
        img = img / np.max(img)
        img = 1 - img

        # plt.pcolor(img, cmap='Wistia')
        # plt.show()

        data.append(img)
    data = torch.tensor(data)\
        .float().unsqueeze(1).to(device)
    dataset = TensorDataset(data, torch.zeros_like(data))
    dataloader = DataLoader(dataset, 1,  shuffle=False)

    for f in os.scandir('models'):
        if f.name != 'LargeWin_4_2_2_3_2_7_7_wd.pt':
            continue
        print(f)
        model = models.load_from_file(f)
        eval_model(model, dataloader)

```

While running the code, it failed right away for me:

![im_02_pred](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\im_02_pred.png)

It predicts a completely wrong direction (which is by the way not the case for all the images). So we have a chance to investigate this case.

### 5.2 Investigation

The easiest part to start is to take a look at the output of the second convolution layer, through the tool we made previously. I only had to replace the part which generates a random image with the code that reads the actual image:

```python
# np.random.seed(42)
# img, lbl = generate_image()
img = cv.imread('data_real/01.png', cv.IMREAD_GRAYSCALE)
img_size = np.min(img.shape)
img = img[:img_size, :img_size]
img = img.astype(float)
img = cv.resize(img, (30, 30))
img = img - np.min(img)
img = img / np.max(img)
img = 1 - img
```

After running the code, got the following result:

![03_filt_num](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\03_filt_num.png)

After some thinking and reminding myself how this system should work, I figured out the source of the problem:

![03_filt_num_scheme](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\03_filt_num_scheme.png)

The image is contained mostly in the central pixels, but the filter used the edge pixels to make the decision. So the easy fix was to cut the real image even more, so those edge pixels after convolution would contain information:

![01_c](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\01_c.png)

The evaluation code gave much better result this time:

![im_02_pred_c](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\im_02_pred_c.png)

But an easy solution is a bit boring, isn't it? A much better solution would be to make the filter care about the center pixels. To do so, we might include this case into the training set (by which I mean to include smaller circles shifted to a side of the image). After training the network, we may hope that our real images will be recognized a lot better.

### 5.6 The fix

So, the plan is to shift the image. But we don't want to shift it too much, so that the point at which the circle is open would remain on the image. Otherwise, the network simply will not train. So we want to shift the image only if it is small enough. 

The part I will be changing is the generate_image() function. The part where the image is being scaled:

```python
    scale_x = np.random.uniform(0.5, 1.1)
    scale_y = np.random.uniform(0.5, 1.1)
```

I will add it a chance to scale even further, say up to 0.2 times original size:

```python
    scale_x = np.random.uniform(0.2, 1.1)
    scale_y = np.random.uniform(0.2, 1.1)
```

After that, I will decide how much I want to shift it:

```python
    shift_x, shift_y = 0, 0
    shift_thresh = 0.5
    if scale_x < shift_thresh:
        shift_x = np.random.uniform(
            -shift_thresh + scale_x, 
            shift_thresh - scale_x
        )
    if scale_y < shift_thresh:
        shift_y = np.random.uniform(
            -shift_thresh + scale_y, 
            shift_thresh - scale_y
        )
```

Here I initialize the shift value to zero, then if my image is scaled to less than 0.5 times its original size - I will change the shift values. The magnitude of this change will be at most "shift_thresh - scale_y", at both sides. This will ensure that the whole information stays inside the image.

Now before we apply this shift, note that there is a rotation transform, and the shift has to be applied after the rotation. Otherwise, the image will be rotated not around the center of the circle (but around the center of the image, which is away from the circle). 

So, here's what the new function looks like:

```python
def generate_image():
    n_circle_pts = 350

    # Parameters randomization
    angle = np.random.uniform(0, 360)
    open_percent = np.random.uniform(10, 40)
    circle_p1 = np.random.uniform(0.7, 1.)
    circle_p2 = np.random.uniform(0.7, 1.)
    scale_x = np.random.uniform(0.2, 1.1)
    scale_y = np.random.uniform(0.2, 1.1)
    shift_x, shift_y = 0, 0
    shift_thresh = 0.5
    if scale_x < shift_thresh:
        shift_x = np.random.uniform(
            -shift_thresh + scale_x,
            shift_thresh - scale_x
        )
    if scale_y < shift_thresh:
        shift_y = np.random.uniform(
            -shift_thresh + scale_y,
            shift_thresh - scale_y
        )

    n_pts_skip = int(n_circle_pts / 100 * open_percent)
    angle_rad = angle / 180 * np.pi
    xs, ys = get_circle(circle_p1, circle_p2, n_circle_pts)
    xs, ys = transform(xs, ys, angle, scale_x, scale_y)
    xs += shift_x
    ys += shift_y
    img = to_image(xs[n_pts_skip // 2:-n_pts_skip // 2],
                   ys[n_pts_skip // 2:-n_pts_skip // 2], 30)

    label = [np.sin(angle_rad), np.cos(angle_rad)]
    return img, label
```

Be careful not to overestimate the 'shift_thresh' value. If it is too large, some images will not include the information needed, and the network will not learn. In my case, the value of 0.8 was too much.

### 5.7 The result

After training the network I got a significantly better result. The second convolution filters started paying attention to the central pixels, and all the real images were recognized (more or less) correctly.

Here is the image of the weights after training:

![fixed_l1](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\fixed_l1.png)

![fixed_l2](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\fixed_l2.png)

These are the filters for the first and second convolutions along with the output for the same real image I used before. The filters have a remarkably better filling of the central pixels.  

Here's what the evaluation script outputs with the newly trained model:

![im_03_pred](C:\Users\vpogribnyi\Documents\Dojo\ML\OpenCircle\v3\images\05_real\im_03_pred.png)

Overall, I'm happy with the model's performance. I think I'm ready to leave it as it is.

## Ourto

This article does not cover many topics it may have covered. For example, the operation of the first convolution layer. Or the connection between the second convolution and the ultimate output may be described in more detail. Research of other architectures and their comparison may have been done. But this article is already too large, so I'm afraid I'll have to leave it as it is (at least for now).

This article was meant to give the readers a better understanding of the neural network operation, a simple example of research and debugging. Hope it did well.

Good luck and happy coding! 