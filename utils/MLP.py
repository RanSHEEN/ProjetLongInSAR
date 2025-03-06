import torch
import torch.nn as nn
from torch.utils.data import Dataset
import utils.activations


class MLP(nn.Module):
    def __init__(self, input_size=1920, hidden_sizes=[1024, 512, 256], output_size=30):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ComplexDataset(Dataset):
    def __init__(self, data):
        """
        data : un tenseur complexe torch de forme (nb d'echantillons, 30, 64) de type torch.complex64
        """
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]

