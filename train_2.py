import os
import torch
import mlflow
import models
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def train_model(model, parameters, dataloader):
    opt = torch.optim.Adam(model.parameters())
    mse = torch.nn.MSELoss()
    with mlflow.start_run(run_name="OpenCircle"):
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

    def run_train_model_small(ch1, ch2, ch3, ch4,
                        ks1, ks2, ks3):
        try:
            parameters = {
                "ch1": ch1,
                'ch2': ch2,
                'ch3': ch3,
                'ch4': ch4,
                'ks1': ks1,
                'ks2': ks2,
                'ks3': ks3
            }
            model = models.SmallWin(parameters).to(device)
            train_model(model, parameters, train_dataloader)
        except Exception as e:
             print(e)

    for ch1 in [2, 4]:
        for ch2 in [2, 4]:
            for ch3 in [2, 4]:
                for ch4 in [2, 4]:
                    for ks1 in [3, 5]:
                        for ks2 in [3, 5]:
                            for ks3 in [3, 5]:
                                run_train_model_small(
                                    ch1, ch2, ch3, ch4,
                                    ks1, ks2, ks3
                                )