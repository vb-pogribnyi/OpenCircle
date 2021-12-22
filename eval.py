import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import models

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
        model = load_from_file(f)
        eval_model(model, dataloader)
