import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


def conv_to_fully_connected(input_size, filter_size, padding, stride):
    return (input_size - filter_size + 2 * padding)/ stride + 1


class NN_0(nn.Module):

    def __init__(self, system_size, dropout, number_of_actions, device):
        super(NN_0, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(64*system_size*system_size, 512)
        self.linear2 = nn.Linear(512, 3) # 0: x, 1: y, 3: z

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class NN_1(nn.Module):

    def __init__(self, system_size, dropout, number_of_actions, device):
        super(NN_1, self).__init__()
        self.kernel_size = 3
        self.conv = nn.Conv2d(2, 1024, kernel_size=self.kernel_size, stride=1)
        self.output = conv_to_fully_connected(system_size, self.kernel_size, 0, 1)
        self.sizes = [int(self.output*self.output*1024), 900, 700, 600, 512, 256]
        self.len_sizes = len(self.sizes)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.linear = [nn.Linear(self.sizes[i-1], self.sizes[i]) for i in range(1, self.len_sizes)]
        self.output = nn.Linear(self.sizes[-1], number_of_actions) 
        self.device = device

    def forward(self, x):
        x = F.relu(self.conv(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        for layer in self.linear:
            layer = layer.to(self.device)
            x = F.relu(layer(x))
            x = self.dropout_layer(x)
        x = self.output(x)
        return x


class NN_2(nn.Module):

    def __init__(self, system_size, dropout, number_of_actions, device):
        super(NN_2, self).__init__()
        self.conv1 = nn.Conv2d(2, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(64*system_size*system_size, 512)        
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 3) # 0: x, 1: y, 3: z

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class NN_3(nn.Module):

    def __init__(self, system_size, dropout, number_of_actions, device):
        super(NN_3, self).__init__()
        self.kernel_size = 3
        self.conv = nn.Conv2d(2, 1024, kernel_size=self.kernel_size, stride=1)
        self.output = conv_to_fully_connected(system_size, self.kernel_size, 0, 1)
        self.sizes = [self.output, 900, 700, 600, 512, 256]
        self.len_sizes = len(self.sizes)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.linear = [nn.Linear(self.sizes[i-1], self.sizes[i]) for i in range(1, self.len_sizes)]
        self.batch_normalization = [nn.BatchNorm1d(self.sizes[i]) for i in range(1, self.len_sizes)]
        self.output = nn.Linear(self.sizes[-1], number_of_actions)

    def forward(self, x):
        x = F.relu(self.conv(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        
        for layer, batch_normalization_layer in zip(self.linear, self.batch_normalization):
            x = F.relu(layer(x))
            x = self.dropout_layer(x)
            x = batch_normalization_layer(x) # doesnt work 
            
        x = self.output(x)
        return x

class NN_4(nn.Module):

    def __init__(self, system_size, dropout, number_of_actions, device):
        super(NN_4, self).__init__()
        self.conv1 = nn.Conv2d(2, 756, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(756, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3, stride=1)
        output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
        self.linear1 = nn.Linear(64*int(output_from_conv)**2, 512)        
        self.linear2 = nn.Linear(512, 3) # 0: x, 1: y, 3: z

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class NN_5(nn.Module):

    def __init__(self, system_size, dropout, number_of_actions, device):
        super(NN_5, self).__init__()
        self.conv1 = nn.Conv2d(2, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1)
        output_from_conv = conv_to_fully_connected(system_size, 3, 0, 1)
        self.linear1 = nn.Linear(64*int(output_from_conv)**2, 128)        
        self.linear2 = nn.Linear(128, 3) # 0: x, 1: y, 3: z

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        n_features = np.prod(x.size()[1:])
        x = x.view(-1, n_features)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
