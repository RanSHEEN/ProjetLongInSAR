import numpy as np
import scipy.linalg as La
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0,
    os.path.join(os.path.dirname(__file__), '../'))

from src.data_generation import simulate_Covariance, phasegeneration, \
    simulate_gaussiandata, simulate_scaledgaussiandata
from src.covariance_estimators import SCM, regul_linear, bandw, SCM_LR, \
    tyler_estimator_covariance, tyler_estimator_covariance_LR, corr_phase, corr_phase_LR
from src.optimization import MM_KL_IPL, MM_LS_IPL, RG_comet_IPL, RG_LS_IPL

from utils.data_gen_func import data_gen, normalize_data
from utils.MLP import MLP, ComplexDataset, MLP_1_activ
from utils.activations import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.loss import complex_mse_loss, weighted_arg_mod_loss
import torch.optim as optim
import matplotlib.pyplot as plt
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_train, output_train = data_gen(size=500, p=10, n=25)
input_val, output_val = data_gen(size=100, p=10, n=25)

# formatage des données pour le MLP
input_train_norm, output_train_norm, real_mean, real_std, imag_mean, imag_std = normalize_data(input_train, output_train)
input_val_norm, output_val_norm, real_mean, real_std, imag_mean, imag_std = normalize_data(input_val, output_val)

input_train_norm = np.array(input_train)
input_val_norm = np.array(input_val)
output_train_norm = np.array(output_train)
output_val_norm = np.array(output_val)      

input_train_tensor_norm = torch.tensor(input_train, dtype=torch.complex64).to(device)
input_val_tensor_norm = torch.tensor(input_val, dtype=torch.complex64).to(device)
output_train_tensor_norm = torch.tensor(output_train, dtype=torch.complex64).to(device)
output_val_tensor_norm = torch.tensor(output_val, dtype=torch.complex64).to(device)


