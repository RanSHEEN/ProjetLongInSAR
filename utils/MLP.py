import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils.activations import Cardioid, CartReLU, ZReLU, ModReLU


class MLP(nn.Module):
    def __init__(self, input_size=1920, hidden_sizes=[1024, 512, 256], output_size=30, activation='cartReLU'):
        super(MLP, self).__init__()
        self.hidden_sizes = hidden_sizes

        # Créer une liste pour stocker les couches du réseau
        layers = []

        # Ajouter la première couche linéaire avec des poids complexes
        layers.append(self._create_complex_linear(input_size, hidden_sizes[0]))

        # Ajouter les couches cachées
        for i in range(1, len(hidden_sizes)):
            layers.append(self._get_activation(activation, hidden_sizes[i-1]))
            layers.append(self._create_complex_linear(hidden_sizes[i-1], hidden_sizes[i]))

        # Ajouter la couche de sortie
        layers.append(self._get_activation(activation, hidden_sizes[-1]))
        layers.append(self._create_complex_linear(hidden_sizes[-1], output_size))

        # Convertir la liste en un ModuleList pour le stockage
        self.layers = nn.ModuleList(layers)

    def _create_complex_linear(self, in_features, out_features):
        # Crée une couche linéaire avec des poids et des biais complexes
        linear = nn.Linear(in_features, out_features)
        linear.weight = nn.Parameter(linear.weight.to(torch.complex64))
        linear.bias = nn.Parameter(linear.bias.to(torch.complex64))
        return linear

    def _get_activation(self, activation, in_features):
        # Retourne la fonction d'activation appropriée
        if activation == 'cartReLU':
            return CartReLU()
        elif activation == 'zReLU':
            return ZReLU()
        elif activation == 'cardioid':
            return Cardioid()
        elif activation == 'modReLU':
            return ModReLU(in_features)
        else:
            raise ValueError(f"Activation '{activation}' is not supported.")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x.unsqueeze(-1)  # Ajouter une dimension singleton pour correspondre à [batch_size, 30, 1]






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

