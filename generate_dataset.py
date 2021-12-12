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
