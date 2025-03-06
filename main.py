from utils.data_gen_func import data_gen
from utils.MLP import MLP, ComplexDataset
from utils.activations import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = MLP()
print(model)

input_train, output_train = data_gen()
input_val, output_val = data_gen(size=100)

# Convertir les listes en tableaux NumPy uniques
input_train = np.array(input_train)  # Convertir en un seul tableau NumPy
input_val = np.array(input_val)      # Convertir en un seul tableau NumPy
output_train = np.array(output_train)  # Convertir en un seul tableau NumPy
output_val = np.array(output_val)      # Convertir en un seul tableau NumPy

# Convertir en tenseurs complexes
input_train_tensor = torch.tensor(input_train, dtype=torch.complex64)
input_val_tensor = torch.tensor(input_val, dtype=torch.complex64)
output_train_tensor = torch.tensor(input_train, dtype=torch.complex64)
output_val_tensor = torch.tensor(input_val, dtype=torch.complex64)

# Passage au format ComplexDataset
in_train_dataset = ComplexDataset(input_train_tensor)
out_train_dataset = ComplexDataset(output_train_tensor)
in_val_dataset = ComplexDataset(input_val_tensor)
out_val_dataset = ComplexDataset(output_val_tensor)

# Création des dataloader
batch_size = 32
in_train_dataloader = DataLoader(in_train_dataset, batch_size=batch_size)
out_train_dataloader = DataLoader(out_train_dataset, batch_size=batch_size)
in_val_dataloader = DataLoader(in_val_dataset, batch_size=batch_size)
out_val_dataloader = DataLoader(out_val_dataset, batch_size=batch_size)
