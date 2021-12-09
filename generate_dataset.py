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


if __name__ == '__main__':
    xs, ys = get_circle(0.4, 0.8, 100)
    xs, ys = transform(xs, ys, 10, 0.8, 1.3)

    plt.subplot(1, 2, 1)
    plt.plot(xs, ys)
    plt.subplot(1, 2, 2)
    plt.plot(xs[15:-15], ys[15:-15])
    plt.gcf().set_size_inches(10, 5)
    plt.show()
