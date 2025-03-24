from utils.data_gen_func import data_gen, normalize_data
import os
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
input_train, output_train, true = data_gen(size=2, p=10, n=25)
print(np.shape(true))
print(np.shape(output_train))

