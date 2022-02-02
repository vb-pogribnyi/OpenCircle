import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("runs.csv")
data = data[[
    'model_type',
    'Status',
    'train_loss',
    'ch1', 'ch2', 'ch3', 'ch4',
    'ks1', 'ks2', 'ks3',
    'pool1_size', 'pool2_size'
]]
print(data.shape)
data = data[data['Status'] == 'FINISHED']
data = data[data['model_type'] == 'SmallWin']
print(data.shape)

print(data.corr(method='pearson')['train_loss'])

plt.scatter(data['ch1'], data['train_loss'],
            label="Ch1", alpha=0.2)
plt.scatter(data['ks2'], data['train_loss'],
            label="Ks2", alpha=0.2)
plt.legend()
plt.show()
