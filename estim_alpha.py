from utils.data_gen_func import data_gen, normalize_data
from utils.MLP import MLP, ComplexDataset, MLP_1_activ
from utils.activations import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from utils.loss import complex_mse_loss, weighted_arg_mod_loss
import torch.optim as optim
import matplotlib.pyplot as plt
import time


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == "__main__":

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculs effectués sur : {device}\n")
    print('-------------------------------------------------\n')

    model_0 = MLP(file='none_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='affsin', alpha=0)
    print('None model \n')
    print(model_0)
    model_0 = model_0.to(device)

    print('\n-------------------------------------------------------------')
    model_0005 = MLP(file='none_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='affsin', alpha=0.005)
    print('None model \n')
    print(model_0005)
    model_0005 = model_0005.to(device)

    print('\n-------------------------------------------------------------')
    model_005 = MLP(file='none_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='affsin', alpha=0.05)
    print('None model \n')
    print(model_005)
    model_005 = model_005.to(device)
    print('\n-------------------------------------------------------------')


    model_05 = MLP(file='none_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='affsin', alpha=0.5)
    print('None model \n')
    print(model_05)
    model_05 = model_05.to(device)
    print('\n-------------------------------------------------------------')


    model_1 = MLP(file='none_hidden_out.txt', hidden_sizes=[2048, 256, 128], activation='affsin', alpha=1)
    print('None model \n')
    print(model_1)
    model_1 = model_1.to(device)
    print('\n-------------------------------------------------------------')


    model_names = ['0', '0.005', '0.05', '0.5', '1']
    model_list = [model_0, model_0005, model_005, model_05, model_1]

    optimizer_0 = optim.Adam(model_0.parameters(), lr=0.001)
    optimizer_0005 = optim.Adam(model_0005.parameters(), lr=0.001)
    optimizer_005 = optim.Adam(model_005.parameters(), lr=0.001)
    optimizer_05 = optim.Adam(model_05.parameters(), lr=0.001)
    optimizer_1 = optim.Adam(model_1.parameters(), lr=0.001)
    optimizer_list = [optimizer_0, optimizer_0005, optimizer_005, optimizer_05, optimizer_1]

    train_losses_0 = []
    val_losses_0 = []
    train_losses_0005 = []
    val_losses_0005 = []
    train_losses_005 = []
    val_losses_005 = []
    train_losses_05 = []
    val_losses_05 = []
    train_losses_1 = []
    val_losses_1 = []
    train_losses = [train_losses_0, train_losses_0005, train_losses_005, train_losses_05, train_losses_1]
    val_losses = [val_losses_0, val_losses_0005, val_losses_005, val_losses_05, val_losses_1]


    # Boucle d'entraînement
    num_epochs = 50
    for model, optimizer, train, val, name in zip(model_list, optimizer_list, train_losses, val_losses, model_names):
        start_mod = time.time()
        print('-------------------------------------------------------------------------------------------------------\n')
        print('Modèle : ', name)
        for epoch in range(num_epochs):
            model.train()  # Passer le modèle en mode entraînement
            train_loss = 0.0
            input_train, output_train = data_gen(size=500, p=10, n=25)
            input_val, output_val = data_gen(size=100, p=10, n=25)

            input_train, output_train = normalize_data(input_train, output_train)
            input_val, output_val = normalize_data(input_val, output_val)

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
        end_mod = time.time()
        exec_mod = end_mod - start_mod
        print(f"Temps d'entraînement du modèle {name} : {exec_mod:.2f} secondes.\n")

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

