import numpy as np
import scipy.linalg as La
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kl_div

import sys
import os
sys.path.insert(0,
    os.path.join(os.path.dirname(__file__), '../'))


from utils.data_gen_func import data_gen, normalize_data, denormalize_w_list
from utils.MLP import MLP
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.loss import complex_mse_loss_np, euclidean_distance_complex_list
import matplotlib.pyplot as plt
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


input_train, output_train, true_train, rho_list = data_gen(size=500, p=10, n=25)
# formatage des données pour le MLP
input_train_norm, output_train_norm, real_mean, real_std, imag_mean, imag_std, original_norms = normalize_data(input_train, output_train)
input_train_norm = np.array(input_train)
output_train_norm = np.array(output_train)
input_train_tensor_norm = torch.tensor(input_train, dtype=torch.complex64).to(device)
output_train_tensor_norm = torch.tensor(output_train, dtype=torch.complex64).to(device)


model_modmvn = MLP(file='compar_mvn.txt', hidden_sizes=[2048, 256, 128], activation='modmvn')
model_modmvn.load_state_dict(torch.load('model_ModMVN.pth'))
model_modmvn.eval()
model_modmvn.to(device)
model_name = 'ModMVN'


with torch.no_grad():
    output_train_pred = model_modmvn(input_train_tensor_norm)

# Déplacer le tenseur vers la CPU et convertir en tableau NumPy
output_train_pred_np = output_train_pred.cpu().numpy()
output_train_pred_np = output_train_pred_np.reshape(200, 10)
# Dénormaliser
output_train_pred_denorm = denormalize_w_list(output_train_pred_np, original_norms)
output_train_np = np.array(output_train)
output_train_reshaped = output_train_np.reshape(200, 10)


mse_loss_denorm = complex_mse_loss_np(output_train_pred_denorm, true_train)
mse_loss_train = complex_mse_loss_np(output_train_reshaped, true_train)
print("Distance MSE complexe entre le vecteur de phase et son estimation par MLP :", mse_loss_denorm)
print("Distance MSE complexe entre le vecteur de phase et son estimation par COFI PL :", mse_loss_train)
mse_losses_denorm = [complex_mse_loss_np(output_train_pred_denorm[i], true_train[i]) for i in range(len(rho_list))]
mse_losses_train = [complex_mse_loss_np(output_train_reshaped[i], true_train[i]) for i in range(len(rho_list))]
# Regrouper les listes avec zip
zipped_lists = list(zip(rho_list, mse_losses_denorm, mse_losses_train))
# Trier les listes en fonction de rho
zipped_lists_sorted = sorted(zipped_lists, key=lambda x: x[0])
# Décompresser les listes triées
rho_list_sorted, mse_losses_denorm_sorted, mse_losses_train_sorted = zip(*zipped_lists_sorted)
# Tracer les pertes MSE en fonction de rho
plt.figure(figsize=(12, 6))
plt.plot(rho_list_sorted, mse_losses_denorm_sorted, label='MSE Loss Denorm', color='blue')
plt.plot(rho_list_sorted, mse_losses_train_sorted, label='MSE Loss Train', color='orange')
plt.xlabel('Rho')
plt.ylabel('MSE Loss')
plt.title('MSE Loss vs Rho (Sorted)')
plt.legend()
plt.show()



distances_denorm = euclidean_distance_complex_list(output_train_pred_denorm, true_train)
mean_denorm = np.mean(distances_denorm)
# Calculer les distances euclidiennes entre true_train et output_train_reshaped
distances_train = euclidean_distance_complex_list(output_train_reshaped, true_train)
mean_train = np.mean(distances_train)
print("Distance euclidienne  moyenne entre le vecteur de phase et son estimation par MLP :", mean_denorm)
print("Distance euclidienne  moyenne entre le vecteur de phase et son estimation par COFI PL :", mean_train)
# Trier les distances en fonction de rho
zipped_lists = list(zip(rho_list, distances_denorm, distances_train))
zipped_lists_sorted = sorted(zipped_lists, key=lambda x: x[0])
# Décompresser les listes triées
rho_list_sorted, distances_denorm_sorted, distances_train_sorted = zip(*zipped_lists_sorted)
# Tracer les distances euclidiennes en fonction de rho
plt.figure(figsize=(12, 6))
plt.plot(rho_list_sorted, distances_denorm_sorted, label='Distance Euclidienne Denorm', color='blue')
plt.plot(rho_list_sorted, distances_train_sorted, label='Distance Euclidienne Train', color='orange')
plt.xlabel('Rho')
plt.ylabel('Distance Euclidienne')
plt.title('Distance Euclidienne vs Rho (Sorted)')
plt.legend()
plt.show()

mse_loss_denorm_bis = complex_mse_loss_np(output_train_pred_denorm, output_train_reshaped)
print("Distance MSE complexe entre MLP et COFI PL :", mse_loss_denorm_bis)

distances_denorm = euclidean_distance_complex_list(output_train_pred_denorm, output_train_reshaped)
mean_denorm = np.mean(distances_denorm)
print("Distance euclidienne  moyenne entre MLP et COFI PL :", mean_denorm)
