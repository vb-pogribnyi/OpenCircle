import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_dataset import generate_image
import models

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

np.random.seed(42)

img, lbl = generate_image()
t_img = torch.tensor(img).float() \
    .unsqueeze(0).unsqueeze(0)

good_models = [
    'LargeWin_4_2_2_3_2_7_7_wd.pt',
    # 'LargeWin_4_2_2_4_2_7_5_wd.pt',
    # 'LargeWin_4_4_2_3_2_7_7_wd.pt',
    # 'LargeWin_4_4_2_4_2_7_5_wd.pt'
]

for f in os.scandir('models'):
    if f.name not in good_models:
        continue
    print(f)
    model = models.load_from_file(f)
    out = model(t_img).detach().reshape(-1).numpy()

    print(out)
    print(lbl)
    plt.pcolor(img, cmap='Wistia')
    plt.show()

    show_weight(model.conv1.weight, model(t_img, 0)[0])
    show_weight(model.conv2.weight, model(t_img, 1)[0])