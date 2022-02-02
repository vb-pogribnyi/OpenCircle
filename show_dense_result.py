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
