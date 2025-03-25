from utils.data_gen_func import data_gen
from utils.MLP import MLP, ComplexDataset
from utils.activations import *
from utils.loss import complex_mse_loss, weighted_arg_mod_loss
from utils import NPYViewer as npyv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

import time

# Fonction pour afficher un patch
def show_patch(patch, index_img, index_patch):
    plt.figure(figsize=(4, 4))
    plt.imshow(patch, cmap='gray', vmin=np.min(patch), vmax=np.max(patch))
    plt.title(f"Image {index_img+1}, Patch {index_patch+1}")
    plt.axis('off')
    plt.show()

# --- Préparer les patches ---
def prepare_patch_input(patch):
    # patch shape: (100, 100, 40)
    flat_patch = patch.flatten()  # (100*100*40,)
    flat_patch = torch.tensor(flat_patch, dtype=torch.complex64).unsqueeze(0)  # (1, N)
    return flat_patch

# --- Visualiser les résultats ---
def show_prediction_map(results, name, grid_shape):
    """
    results: liste de scalaires de sortie
    grid_shape: (H_patches, W_patches)
    """
    result_array = np.array(results).squeeze()  # (N,)
    result_image = result_array.reshape(grid_shape)  # exemple: (36, 167) selon ton découpage
    
    plt.figure(figsize=(10, 6))
    plt.imshow(result_image, cmap='jet',)
    plt.title(f"Résultat du MLP {name}")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    #%% Entraînement des modèles
    # # Afficher les 40 premiers patches
    # for i in range(num_layers):
    #     current_img = data[:, :, i]

    #     # Découper en patchs de 100x100
    #     patch_id = 0
    #     for y in range(0, height, patch_size):
    #         for x in range(0, width, patch_size):
    #             # S'assurer que le patch reste dans les dimensions
    #             if y + patch_size <= height and x + patch_size <= width:
    #                 patch = current_img[y:y+patch_size, x:x+patch_size]
    #                 show_patch(patch, i, patch_id)
    #                 patch_id += 1

    # start_time = time.time() # Mesurer le temps d'exécution
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('Calculs effectués sur : ', device)

    # # Création des modèles
    # model_mod = MLP(file='mod_hidden_out.txt', hidden_sizes=[2048, 4096, 1024, 512, 256, 128], activation='modReLU')
    # print('ModReLU model')
    # print(model_mod)
    # model_mod = model_mod.to(device)
    # print('\n-------------------------------------------------------------')

    # model_affsin = MLP(file='affsin_hidden_out.txt', hidden_sizes=[2048, 4096, 1024, 512, 256, 128], activation='affsin')
    # print('AffSin model')
    # print(model_affsin)
    # model_affsin = model_affsin.to(device)
    # print('\n-------------------------------------------------------------')
 
    # model_names = ['ModReLU', 'AffSin']
    # model_list = [model_mod, model_affsin]

    # # Optimiseurs
    # optimizer_mod = optim.SGD(model_mod.parameters(), lr=0.1)
    # optimizer_affsin = optim.SGD(model_affsin.parameters(), lr=0.0001)
    # optimizer_list = [optimizer_mod, optimizer_affsin]

    # # Pertes
    # train_losses_card = []
    # val_losses_card = []
    # train_losses_z = []
    # val_losses_z = []
    # train_losses_mod = []
    # val_losses_mod = []
    # train_losses_affsin = []
    # val_losses_affsin = []
    # # train_losses = [train_losses_card, train_losses_z, train_losses_mod]
    # # val_losses = [val_losses_card, val_losses_z, val_losses_mod]
    # train_losses = [train_losses_mod, train_losses_affsin]
    # val_losses = [val_losses_mod, val_losses_affsin]

    # # Boucle d'entraînement
    # num_epochs = 100
    # for model, optimizer, train, val, name in zip(model_list, optimizer_list, train_losses, val_losses, model_names):
    #     print('-------------------------------------------------------------------------------------------------------\n')
    #     print('Modèle : ', name)
    #     for epoch in range(num_epochs):
    #         model.train()  # Passer le modèle en mode entraînement
    #         train_loss = 0.0
    #         input_train, output_train = data_gen(size=500, p=10, n=25)
    #         input_val, output_val = data_gen(size=100, p=10, n=25)

    #         # Convertir les listes en tableaux NumPy uniques
    #         input_train = np.array(input_train)  # Convertir en un seul tableau NumPy
    #         input_val = np.array(input_val)      # Convertir en un seul tableau NumPy
    #         output_train = np.array(output_train)  # Convertir en un seul tableau NumPy
    #         output_val = np.array(output_val)      # Convertir en un seul tableau NumPy

    #         # Convertir en tenseurs complexes
    #         input_train_tensor = torch.tensor(input_train, dtype=torch.complex64).to(device)
    #         input_val_tensor = torch.tensor(input_val, dtype=torch.complex64).to(device)
    #         output_train_tensor = torch.tensor(output_train, dtype=torch.complex64).to(device)
    #         output_val_tensor = torch.tensor(output_val, dtype=torch.complex64).to(device)

    #         # Passage au format ComplexDataset
    #         in_train_dataset = ComplexDataset(input_train_tensor)
    #         out_train_dataset = ComplexDataset(output_train_tensor)
    #         in_val_dataset = ComplexDataset(input_val_tensor)
    #         out_val_dataset = ComplexDataset(output_val_tensor)
    #         # Passage au format ComplexDataset
    #         in_train_dataset = ComplexDataset(input_train_tensor)
    #         out_train_dataset = ComplexDataset(output_train_tensor)
    #         in_val_dataset = ComplexDataset(input_val_tensor)
    #         out_val_dataset = ComplexDataset(output_val_tensor)

    #         # Création des dataloader
    #         batch_size = 32
    #         in_train_dataloader = DataLoader(in_train_dataset, batch_size=batch_size)
    #         out_train_dataloader = DataLoader(out_train_dataset, batch_size=batch_size)
    #         in_val_dataloader = DataLoader(in_val_dataset, batch_size=batch_size)
    #         out_val_dataloader = DataLoader(out_val_dataset, batch_size=batch_size)
    #         # Itérer sur les données d'entraînement
    #         for inputs, targets in zip(in_train_dataloader, out_train_dataloader):
    #             # inputs, targets = inputs.to(device), targets.to(device)
    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             loss = complex_mse_loss(outputs, targets)
    #             loss.backward()
    #             optimizer.step()
    #             train_loss += loss.item()
    #         # Création des dataloader
    #         batch_size = 32
    #         in_train_dataloader = DataLoader(in_train_dataset, batch_size=batch_size)
    #         out_train_dataloader = DataLoader(out_train_dataset, batch_size=batch_size)
    #         in_val_dataloader = DataLoader(in_val_dataset, batch_size=batch_size)
    #         out_val_dataloader = DataLoader(out_val_dataset, batch_size=batch_size)
    #         # Itérer sur les données d'entraînement
    #         for inputs, targets in zip(in_train_dataloader, out_train_dataloader):
    #             # inputs, targets = inputs.to(device), targets.to(device)
    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             loss = complex_mse_loss(outputs, targets)
    #             loss.backward()
    #             optimizer.step()
    #             train_loss += loss.item()

    #         avg_train_loss = train_loss / len(in_train_dataloader)
    #         train.append(avg_train_loss)
    #         avg_train_loss = train_loss / len(in_train_dataloader)
    #         train.append(avg_train_loss)

    #         # Validation
    #         model.eval()  # Passer le modèle en mode évaluation
    #         val_loss = 0.0
    #         with torch.no_grad():
    #             for inputs, targets in zip(in_val_dataloader, out_val_dataloader):
    #                 # inputs, targets = inputs.to(device), targets.to(device)
    #                 outputs = model(inputs)
    #                 loss = complex_mse_loss(outputs, targets)
    #                 val_loss += loss.item()
    #         # Validation
    #         model.eval()  # Passer le modèle en mode évaluation
    #         val_loss = 0.0
    #         with torch.no_grad():
    #             for inputs, targets in zip(in_val_dataloader, out_val_dataloader):
    #                 # inputs, targets = inputs.to(device), targets.to(device)
    #                 outputs = model(inputs)
    #                 loss = complex_mse_loss(outputs, targets)
    #                 val_loss += loss.item()

    #         avg_val_loss = val_loss / len(in_val_dataloader)
    #         val.append(avg_val_loss)
    #         avg_val_loss = val_loss / len(in_val_dataloader)
    #         val.append(avg_val_loss)

    #         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # # Mesurer le temps d'exécution
    # end_time = time.time()
    # exec_time = end_time - start_time
    # print(f"Temps d'exécution total (entraînement de {len[model_list]} modèles sur {num_epochs} epochs) : {exec_time:.2f} secondes")
    
    # # Tracer les pertes
    # plt.figure(figsize=(12, 6))
    # for i, (train_loss, val_loss, model_name) in enumerate(zip(train_losses, val_losses, model_names)):
    #     plt.plot(train_loss, label=f'Train Loss ({model_name})')
    #     plt.plot(val_loss, label=f'Validation Loss ({model_name})', linestyle='--')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss for Different Activations')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    #%% Application des modeles
    # Charger le fichier NPY
    fichier_npy = 'data-mexico/phases.npy'  # Assurez-vous du bon chemin d'accès
    data = np.load(fichier_npy, mmap_mode='r')  # Mode lecture sans tout charger en mémoire

    # Vérification de la forme des données
    print("Shape des données:", data.shape)  # Devrait être (3631, 16702, 40)

    # # Boucle pour afficher les 40 images
    # for i in range(40):  # Parcourt les 40 couches
    #     data_2d = data[:, :, i]
        
    #     # Normalisation des valeurs entre 0 et 255
    #     # data_scaled = 255 * (data_2d - np.min(data_2d)) / (np.max(data_2d) - np.min(data_2d))
    #     # data_scaled = np.uint8(data_scaled)  # Conversion en entier 8 bits
        
    #     # Égalisation d'histogramme
    #     # data_eq = cv2.equalizeHist(data_scaled)
        
    #     plt.figure(figsize=(10, 6))
    #     plt.imshow(data_2d, cmap='gray', aspect='auto')
    #     # plt.colorbar(label="Valeurs")
    #     plt.title(f"Visualisation de la couche {i+1}")
    #     plt.xlabel("Index X")
    #     plt.ylabel("Index Y")
    #     plt.show()

    # Paramètres des patchs
    patch_size = 5
    height, width, num_layers = data.shape

    # Liste pour stocker les patches 3D : (100, 100, 40)
    patches = []

    # Découpe spatiale : chaque patch contient toutes les couches temporelles
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            # Extraire le patch (5, 5, 10)
            patch = data[y:y+patch_size, x:x+patch_size, 15:25]  # garde toutes les couches temporelles
            patches.append(patch)

    print(f"Nombre total de patches 3D extraits : {len(patches)}")
    print(f"Exemple de forme d'un patch : {patches[0].shape}")

    # # Afficher les patches
    # for i in range(num_layers):
    #     current_img = data[:, :, i]

    #     # Découper en patchs de 100x100
    #     patch_id = 0
    #     for y in range(0, height, patch_size):
    #         for x in range(0, width, patch_size):
    #             # S'assurer que le patch reste dans les dimensions
    #             if y + patch_size <= height and x + patch_size <= width:
    #                 patch = current_img[y:y+patch_size, x:x+patch_size]
    #                 show_patch(patch, i, patch_id)
    #                 patch_id += 1

    start_time = time.time() # Mesurer le temps d'exécution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Calculs effectués sur : ', device)

     # Dictionnaire des modèles à charger
    model_names = ['AffSin', 'Cardioid', 'CartReLU', 'CartTanh', 'ModMVN', 'ModReLU', 'None', 'ZReLU']
    model_files = {
        'AffSin': 'model_AffSin.pth',
        'Cardioid': 'model_Cardioid.pth',
        'CartReLU': 'model_CartReLU.pth',
        'CartTanh': 'model_CartTanh.pth',
        'ModMVN': 'model_ModMVN.pth',
        'ModReLU': 'model_ModReLU.pth',
        'None': 'model_None.pth',
        'ZReLU': 'model_ZReLU.pth'
    }
    activations = {
        'AffSin': 'affsin',
        'Cardioid': 'cardioid',
        'CartReLU': 'cartReLU',
        'CartTanh': 'tanh',
        'ModMVN': 'modmvn',
        'ModReLU': 'modReLU',
        'None': None,
        'ZReLU': 'zReLU'
    }

    model_list = []
    results_by_model = {name: [] for name in model_names}

    # for name in model_names:
    name = 'AffSin'
    model = MLP(file=None, hidden_sizes=[2048, 256, 128], activation=activations[name])
    print(f'{name} model')
    print(model)
    model.load_state_dict(torch.load(model_files[name], map_location=device))
    model = model.to(device)
    model_list.append(model)
    print('\n-------------------------------------------------------------')

    with torch.no_grad():
        # If the memory is sufficient, you can predict all models at once
        # for name, model in zip(model_names, model_list):
        print(f"Prédiction avec {name}...")
        model.eval()
        for patch in patches:
            input_tensor = prepare_patch_input(patch).to(device)
            output = model(input_tensor)
            scalar_output = output.abs().mean().item()
            results_by_model[name].append(scalar_output)
            # results_by_model[name].append(output.cpu().numpy())
        # Vérification du nombre de résultats
        print(f"{name} → {len(results_by_model[name])} patches prédits")

    # Déduction de la grille à partir du nombre total de patches
    total_patches = len(results_by_model[name])
    w_patches = (width - patch_size + 1) // patch_size + 1
    h_patches = total_patches // w_patches  # Division entière
    print(f"Grille prévue : {h_patches} x {w_patches} = {h_patches * w_patches} patches")

    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Temps d'exécution total (entraînement de {name} modèles : {exec_time:.2f} secondes")
    
    # for name in model_names:
    show_prediction_map(results_by_model[name], name, grid_shape=(h_patches, w_patches))

    
