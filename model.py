import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu as lRelu
from torch import tanh


class LanguageNet(nn.Module):

    def __init__(self, vector_size, word_length, num_languages):
        super(LanguageNet, self).__init__()
        input_size = vector_size * word_length
        print(f'vector_size * word_length = {input_size}')
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, num_languages)
        # self.conv1 = nn.Conv2d(1, 3, kernel_size=(vector_size, 3), stride=stride, padding=padding)

        # for p in self.parameters():
            # print(p)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = lRelu(self.fc1(x))
        x = lRelu(self.fc2(x))
        x = lRelu(self.fc3(x))
        x = lRelu(self.fc4(x))
        x = self.fc5(x)

        return x


class ConvLanguageNet(nn.Module):

    def __init__(self, vector_size, word_length, num_languages):
        super(ConvLanguageNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (3, vector_size))
        self.conv2 = nn.Conv2d(64, 1, (3, 1))
        self.fc1 = nn.Linear(word_length - 4, 30)
        self.fc2 = nn.Linear(30, 30)
        self.out = nn.Linear(30, num_languages)
        # self.conv1 = nn.Conv2d(1, 3, kernel_size=(vector_size, 3), stride=stride, padding=padding)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.unsqueeze(1)
        x = lRelu(self.conv1(x))
        # print(x.size())
        x = lRelu(self.conv2(x))
        # print(x.size())
        x = x.squeeze()
        x = lRelu(self.fc1(x))
        x = lRelu(self.fc2(x))
        x = self.out(x)

        return x
