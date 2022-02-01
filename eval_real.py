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
