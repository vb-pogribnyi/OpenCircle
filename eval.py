import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import models


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
        model = models.load_from_file(f)
        eval_model(model, dataloader)
