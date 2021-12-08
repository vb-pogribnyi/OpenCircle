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
