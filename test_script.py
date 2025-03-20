from utils.data_gen_func import data_gen, normalize_data
import os
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
input_train, output_train = data_gen(size=2, p=10, n=25)

input_train = np.array(input_train)  # Convertir en un seul tableau NumPy
print(input_train)
input_train, input_train_mean, input_train_std = normalize_data(input_train)
output_train, output_train_mean, output_train_std = normalize_data(output_train)
print(f"input norm {input_train}")
print(input_train_mean)
print(input_train_std)
