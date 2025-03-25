from utils.data_gen_func import data_gen, normalize_data
from utils.MLP import MLP, ComplexDataset, MLP_1_activ
from utils.activations import *
from utils.loss import complex_mse_loss, weighted_arg_mod_loss
from utils import NPYViewer as npyv

import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

import time
import cv2  # OpenCV pour l'égalisation d'histogramme

# Fonction pour afficher un patch
def show_patch(patch, index_img, index_patch):
    plt.figure(figsize=(4, 4))
    plt.imshow(patch, cmap='gray', vmin=np.min(patch), vmax=np.max(patch))
    plt.title(f"Image {index_img+1}, Patch {index_patch+1}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    #%% Visualisation des données
    # npyv.openNPY_CLI_noGUI('data-mexico\phases.npy')
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
    patch_size = 100
    height, width, num_layers = data.shape

    # Liste pour stocker les patches 3D : (100, 100, 40)
    patches = []

    # Découpe spatiale : chaque patch contient toutes les couches temporelles
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            # Extraire le patch (100, 100, 40)
            patch = data[y:y+patch_size, x:x+patch_size, :]  # garde toutes les couches temporelles
            patches.append(patch)

    print(f"Nombre total de patches 3D extraits : {len(patches)}")
    print(f"Exemple de forme d'un patch : {patches[0].shape}")

    # Boucle sur chaque image
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


<<<<<<< HEAD
    start_time = time.time() # Mesurer le temps d'exécution
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Calculs effectués sur : ', device)

    # Création des modèles
    model_mod = MLP(file='mod_hidden_out.txt', hidden_sizes=[2048, 4096, 1024, 512, 256, 128], activation='modReLU')
    print('ModReLU model')
    print(model_mod)
    model_mod = model_mod.to(device)
=======
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == "__main__":

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculs effectués sur : {device}\n")
    print('-------------------------------------------------\n')

    model_none = MLP(file='none_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='')
    print('None model \n')
    print(model_none)
    model_none = model_none.to(device)

    print('\n-------------------------------------------------------------')
    model_cart = MLP(file='cartReLU_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='cartReLU')
    print('CartReLU model \n')
    print(model_cart)
    model_card = model_cart.to(device)

    print('\n-------------------------------------------------------------')
    model_card = MLP(file='cardioid_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='cardioid')
    print('Cardioid model \n')
    print(model_card)
    model_card = model_card.to(device)
>>>>>>> origin/felix
    print('\n-------------------------------------------------------------')

    model_affsin = MLP(file='affsin_hidden_out.txt', hidden_sizes=[2048, 4096, 1024, 512, 256, 128], activation='affsin')
    print('AffSin model')
    print(model_affsin)
    model_affsin = model_affsin.to(device)
    print('\n-------------------------------------------------------------')
 
    model_names = ['ModReLU', 'AffSin']
    model_list = [model_mod, model_affsin]

<<<<<<< HEAD
    # Optimiseurs
    optimizer_mod = optim.SGD(model_mod.parameters(), lr=0.1)
    optimizer_affsin = optim.SGD(model_affsin.parameters(), lr=0.0001)
    optimizer_list = [optimizer_mod, optimizer_affsin]
=======
    model_z = MLP(file='z_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='zReLU')
    print('ZReLU model')
    print(model_z)
    model_z = model_z.to(device)
    print('\n-------------------------------------------------------------')
>>>>>>> origin/felix

    # Pertes
    train_losses_card = []
    val_losses_card = []
    train_losses_z = []
    val_losses_z = []
    train_losses_mod = []
    val_losses_mod = []
    train_losses_affsin = []
    val_losses_affsin = []
    # train_losses = [train_losses_card, train_losses_z, train_losses_mod]
    # val_losses = [val_losses_card, val_losses_z, val_losses_mod]
    train_losses = [train_losses_mod, train_losses_affsin]
    val_losses = [val_losses_mod, val_losses_affsin]

<<<<<<< HEAD
    # Boucle d'entraînement
    num_epochs = 100
    for model, optimizer, train, val, name in zip(model_list, optimizer_list, train_losses, val_losses, model_names):
        print('-------------------------------------------------------------------------------------------------------\n')
        print('Modèle : ', name)
        for epoch in range(num_epochs):
            model.train()  # Passer le modèle en mode entraînement
            train_loss = 0.0
            input_train, output_train = data_gen(size=500, p=10, n=25)
            input_val, output_val = data_gen(size=100, p=10, n=25)

            # Convertir les listes en tableaux NumPy uniques
            input_train = np.array(input_train)  # Convertir en un seul tableau NumPy
            input_val = np.array(input_val)      # Convertir en un seul tableau NumPy
            output_train = np.array(output_train)  # Convertir en un seul tableau NumPy
            output_val = np.array(output_val)      # Convertir en un seul tableau NumPy

            # Convertir en tenseurs complexes
            input_train_tensor = torch.tensor(input_train, dtype=torch.complex64).to(device)
            input_val_tensor = torch.tensor(input_val, dtype=torch.complex64).to(device)
            output_train_tensor = torch.tensor(output_train, dtype=torch.complex64).to(device)
            output_val_tensor = torch.tensor(output_val, dtype=torch.complex64).to(device)

            # Passage au format ComplexDataset
            in_train_dataset = ComplexDataset(input_train_tensor)
            out_train_dataset = ComplexDataset(output_train_tensor)
            in_val_dataset = ComplexDataset(input_val_tensor)
            out_val_dataset = ComplexDataset(output_val_tensor)

=======
    model_mod = MLP(file='mod_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='modReLU')
    print('ModReLU model')
    print(model_mod)
    model_mod = model_mod.to(device)
    print('\n-------------------------------------------------------------')

    model_affsin = MLP(file='affsin_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='affsin')
    print('AffSin model')
    print(model_affsin)
    model_affsin = model_affsin.to(device)
    print('\n-------------------------------------------------------------')

    model_tanh = MLP(file='carttanh_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='tanh')
    print('CartTanh model')
    print(model_tanh)
    model_tanh = model_tanh.to(device)
    print('\n-------------------------------------------------------------')

    model_modmvn = MLP(file='modmvn_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='modmvn')
    print('ModMVN model')
    print(model_modmvn)
    model_modmvn = model_modmvn.to(device)
    print('\n-------------------------------------------------------------')

    model_names = ['None', 'CartReLU', 'Cardioid', 'ZReLU', 'ModReLU', 'AffSin', 'CartTanh', 'ModMVN']
    model_list = [model_none, model_cart, model_card, model_z, model_mod, model_affsin, model_tanh, model_modmvn]

    optimizer_none = optim.Adam(model_none.parameters(), lr=0.001)
    optimizer_cart = optim.Adam(model_cart.parameters(), lr=0.001)
    optimizer_card = optim.Adam(model_card.parameters(), lr=0.001)
    optimizer_z = optim.Adam(model_z.parameters(), lr=0.001)
    optimizer_mod = optim.Adam(model_mod.parameters(), lr=0.001)
    optimizer_affsin = optim.Adam(model_affsin.parameters(), lr=0.001)
    optimizer_tanh = optim.Adam(model_tanh.parameters(), lr = 0.001)
    optimizer_modmvn = optim.Adam(model_modmvn.parameters(), lr = 0.001)
    optimizer_list = [optimizer_none, optimizer_cart, optimizer_card, optimizer_z, optimizer_mod, optimizer_affsin, optimizer_tanh, optimizer_modmvn]

    train_losses_none = []
    val_losses_none = []
    train_losses_cart = []
    val_losses_cart = []
    train_losses_card = []
    val_losses_card = []
    train_losses_z = []
    val_losses_z = []
    train_losses_mod = []
    val_losses_mod = []
    train_losses_affsin = []
    val_losses_affsin = []
    train_losses_tanh = []
    val_losses_tanh = []
    train_losses_modmvn = []
    val_losses_modmvn = []
    train_losses = [train_losses_none, train_losses_cart, train_losses_card, train_losses_z, train_losses_mod, train_losses_affsin, train_losses_tanh, train_losses_modmvn]
    val_losses = [val_losses_none, val_losses_cart, val_losses_card, val_losses_z, val_losses_mod, val_losses_affsin, val_losses_tanh, val_losses_modmvn]


    # Boucle d'entraînement
    num_epochs = 100
    for model, optimizer, train, val, name in zip(model_list, optimizer_list, train_losses, val_losses, model_names):
        start_mod = time.time()
        print('-------------------------------------------------------------------------------------------------------\n')
        print('Modèle : ', name)
        for epoch in range(num_epochs):
            model.train()  # Passer le modèle en mode entraînement
            train_loss = 0.0
            input_train, output_train = data_gen(size=500, p=10, n=25)
            input_val, output_val = data_gen(size=100, p=10, n=25)

            input_train, output_train, _, _, _, _ = normalize_data(input_train, output_train)
            input_val, output_val, _, _, _, _ = normalize_data(input_val, output_val)

            # Convertir les listes en tableaux NumPy uniques
            input_train = np.array(input_train)  # Convertir en un seul tableau NumPy
            input_val = np.array(input_val)      # Convertir en un seul tableau NumPy
            output_train = np.array(output_train)  # Convertir en un seul tableau NumPy
            output_val = np.array(output_val)      # Convertir en un seul tableau NumPy

            
            # Convertir en tenseurs complexes
            input_train_tensor = torch.tensor(input_train, dtype=torch.complex64).to(device)
            input_val_tensor = torch.tensor(input_val, dtype=torch.complex64).to(device)
            output_train_tensor = torch.tensor(output_train, dtype=torch.complex64).to(device)
            output_val_tensor = torch.tensor(output_val, dtype=torch.complex64).to(device)

            # Passage au format ComplexDataset
            in_train_dataset = ComplexDataset(input_train_tensor)
            out_train_dataset = ComplexDataset(output_train_tensor)
            in_val_dataset = ComplexDataset(input_val_tensor)
            out_val_dataset = ComplexDataset(output_val_tensor)

>>>>>>> origin/felix
            # Création des dataloader
            batch_size = 32
            in_train_dataloader = DataLoader(in_train_dataset, batch_size=batch_size)
            out_train_dataloader = DataLoader(out_train_dataset, batch_size=batch_size)
            in_val_dataloader = DataLoader(in_val_dataset, batch_size=batch_size)
            out_val_dataloader = DataLoader(out_val_dataset, batch_size=batch_size)
            # Itérer sur les données d'entraînement
            for inputs, targets in zip(in_train_dataloader, out_train_dataloader):
                # inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = complex_mse_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(in_train_dataloader)
            train.append(avg_train_loss)

            # Validation
            model.eval()  # Passer le modèle en mode évaluation
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in zip(in_val_dataloader, out_val_dataloader):
                    # inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = complex_mse_loss(outputs, targets)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(in_val_dataloader)
            val.append(avg_val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
<<<<<<< HEAD
=======
        end_mod = time.time()
        exec_mod = end_mod - start_mod
        print(f"Temps d'entraînement du modèle {name} : {exec_mod:.2f} secondes.\n")
        
        model_path = f'model_{name}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"Modèle sauvegardé à {model_path}")

    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Temps d'exécution total (entraînement de {len(model_list)} modèles sur {num_epochs} epochs) : {exec_time:.2f} secondes")
    # Tracer les pertes
    plt.figure(figsize=(12, 6))

    # Tracer les courbes pour chaque modèle
    for i, (train_loss, val_loss, model_name) in enumerate(zip(train_losses, val_losses, model_names)):
        plt.plot(train_loss, label=f'Train Loss ({model_name})')
        plt.plot(val_loss, label=f'Validation Loss ({model_name})', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Different Activations')
    plt.legend()
    plt.grid(True)
    plt.show()
>>>>>>> origin/felix

    # Mesurer le temps d'exécution
    end_time = time.time()
    exec_time = end_time - start_time
    print(f"Temps d'exécution total (entraînement de {len[model_list]} modèles sur {num_epochs} epochs) : {exec_time:.2f} secondes")
    
    # Tracer les pertes
    plt.figure(figsize=(12, 6))
    for i, (train_loss, val_loss, model_name) in enumerate(zip(train_losses, val_losses, model_names)):
        plt.plot(train_loss, label=f'Train Loss ({model_name})')
        plt.plot(val_loss, label=f'Validation Loss ({model_name})', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for Different Activations')
    plt.legend()
    plt.grid(True)
    plt.show()
