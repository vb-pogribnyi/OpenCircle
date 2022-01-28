import os
import torch
import numpy as np
import models
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

np.random.seed(42)
img, lbl = generate_image()

fig, ax = plt.subplots(1, 2)
show_matrix(img, ax[0])
show_matrix(img[5:15, 5:15], ax[1])
plt.show()

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
