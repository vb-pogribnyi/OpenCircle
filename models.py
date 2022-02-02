import torch
from generate_dataset import generate_image

class LargeWin(torch.nn.Module):
    def __init__(self, params):
        img_size = 30
        ch1, ch2 = params['ch1'], params['ch2']
        ch3 = params['ch3']
        pool1_size = params['pool1_size']
        pool2_size = params['pool2_size']
        ks1, ks2 = params['ks1'], params['ks2']
        super(LargeWin, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, ch1, ks1)
        self.conv2 = torch.nn.Conv2d(ch1, ch2, ks2)
        img_size1 = (img_size - ks1 // 2 * 2) // pool1_size
        img_size2 = (img_size1 - ks2 // 2 * 2) // pool2_size
        ch_in = img_size2 ** 2 * ch2
        if ch_in > 4 or img_size2 <= 0:
            raise Exception('Crazy input')
        self.dense1 = torch.nn.Linear(ch_in, ch3)
        self.dense2 = torch.nn.Linear(ch3, 2)
        self.pool1 = torch.nn.AvgPool2d(pool1_size)
        self.pool2 = torch.nn.AvgPool2d(pool2_size)

    def forward(self, x, out_layer=-1):
        x = self.conv1(x)
        if out_layer == 11:
            return x
        x = self.pool1(x)
        x = torch.tanh(x)
        if out_layer == 12:
            return x

        x = self.conv2(x)
        if out_layer == 21:
            return x
        x = self.pool2(x)
        if out_layer == 22:
            return x
        x = torch.tanh(x)

        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)

        return torch.tanh(x)

class SmallWin(torch.nn.Module):
    def __init__(self, params):
        ch1, ch2 = params['ch1'], params['ch2']
        ch3, ch4 = params['ch3'], params['ch4']
        ks1, ks2 = params['ks1'], params['ks2']
        ks3 = params['ks3']
        img_size = 30
        pool_size = 2
        super(SmallWin, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, ch1, ks1)
        self.conv2 = torch.nn.Conv2d(ch1, ch2, ks2)
        self.conv3 = torch.nn.Conv2d(ch2, ch3, ks3)

        img_size1 = (img_size - ks1 // 2 * 2) // pool_size
        img_size2 = (img_size1 - ks2 // 2 * 2) // pool_size
        img_size3 = (img_size2 - ks3 // 2 * 2) // pool_size
        ch_in = img_size3 ** 2 * ch3
        if ch_in > 4 or img_size3 <= 0:
            raise Exception('Crazy input')
        self.dense1 = torch.nn.Linear(ch_in, ch4)
        self.dense2 = torch.nn.Linear(ch4, 2)
        self.pool = torch.nn.AvgPool2d(pool_size)

    def forward(self, x, out_layer=-1):
        x = self.conv1(x)
        if out_layer == 0:
            return x
        x = self.pool(x)
        x = torch.tanh(x)

        x = self.conv2(x)
        if out_layer == 1:
            return x
        x = self.pool(x)
        x = torch.tanh(x)

        x = self.conv3(x)
        if out_layer == 2:
            return x
        x = self.pool(x)
        x = torch.tanh(x)

        x = x.reshape([x.shape[0], -1])
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)

        return torch.tanh(x)

def load_from_file(f):
    name_parts = f.name.split('.')[0].split('_')
    model_class_name = name_parts[0]
    if model_class_name == 'LargeWin':
        model_class = LargeWin
    elif model_class_name == 'SmallWin':
        model_class = SmallWin
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

if __name__ == '__main__':
    img, label = generate_image()
    print(img.shape)
    modelLargeWin = LargeWin({
        "ch1": 4, 'ch2': 4, 'ch3': 4,
        'pool1_size': 4, 'pool2_size': 2,
        'ks1': 7, 'ks2': 5
    })
    modelSmallWin = SmallWin({
        "ch1": 4, 'ch2': 4, 'ch3': 4, 'ch4': 4,
        'ks1': 3, 'ks2': 3, 'ks3': 5
    })

    img_t = torch.tensor(img).float()
    img_t = img_t.unsqueeze(0).unsqueeze(0)
    outLargeWin = modelLargeWin(img_t)
    print('LargeWin', outLargeWin)
    outSmallWin = modelSmallWin(img_t)
    print('SmallWin', outSmallWin)
