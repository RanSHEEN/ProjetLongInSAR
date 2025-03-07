import torch
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from torch.utils.data import DataLoader

from cvnn.losses import ComplexMeanSquareError

import data_gen
import neural_network

import time


if __name__ == "__main__":
    start_time = time.time()

    rho_list = np.random.uniform(low=0.96, high=0.99, size=200)
    p = 30 # data size
    nu = 0 # scale parameter of K-distributed distribution (0 if Gaussian)
    b = 3 # bandwidth parameter
    alpha = 0.5 # coefficient regularization
    rank = 1 # rank of the covariance matrix (p if full-rank)
    phasechoice='random'#,maxphase 
    cost = 'LS' # cost function: LS, KL or WLS
    estimator = 'PO' # estimator : 'SCM', 'Tyler' or 'PO' 
    regul = 'SK' # regularization: False, LR, SK, BW 
    
    X_list, Sigma_tilde_list, w_theta_list = data_gen.generate_data(rho_list, p, nu, b, alpha, rank, phasechoice, cost, estimator, regul)
    
    # Définition du ratio d'entraînement / validation
    train_ratio = 0.8  # 80% train, 20% validation
    train_size = int(train_ratio * len(X_list))
    val_size = len(X_list) - train_size

    # Division des datasets
    X_train_data, X_val_data = random_split(X_list, [train_size, val_size])
    y_train_data, y_val_data = random_split(w_theta_list, [train_size, val_size])

    # Convertir les listes en tableaux NumPy uniques
    X_train_array = np.array(X_train_data)  # Convertir en un seul tableau NumPy
    y_train_array = np.array(y_train_data)  # Convertir en un seul tableau NumPy
    X_val_data = np.array(X_val_data)  # Convertir en un seul tableau NumPy
    y_val_data = np.array(y_val_data)  # Convertir en un seul tableau NumPy

    # Convertir en tenseurs complexes
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.complex64)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.complex64)
    X_val_tensor = torch.tensor(X_val_data, dtype=torch.complex64)
    y_val_tensor = torch.tensor(y_val_data, dtype=torch.complex64)

    # Passage au format ComplexDataset
    X_train = data_gen.ComplexDataset(X_train_tensor)
    y_train = data_gen.ComplexDataset(y_train_tensor)
    X_val = data_gen.ComplexDataset(X_val_tensor)
    y_val = data_gen.ComplexDataset(y_val_tensor)

    # Paramètres du modèle
    n = 64 # number of samples
    # input_dim = p * n + p * p  # X est de taille (p, n) et Sigma_tilde de taille (p, p)
    input_dim = n
    hidden_dim = 128  # Nombre de neurones cachés
    output_dim = p  # Sortie (ex. estimation de covariance)

    model = neural_network.NeuralNetwork(input_dim, hidden_dim, output_dim) 
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=ComplexMeanSquareError())
    # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=neural_network.custom_complex_mse)
    # model.compile(optimizer=neural_network.ComplexSGD(learning_rate=0.001), loss=ComplexMeanSquareError())

    # Création des DataLoaders
    epochs = 1000
    batch_size = 32
    train_losses = []
    val_losses = []
    X_train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    y_train_dataloader = DataLoader(y_train, batch_size=batch_size, shuffle=True)
    X_val_dataloader = DataLoader(X_val, batch_size=batch_size)
    y_val_dataloader = DataLoader(y_val, batch_size=batch_size)

    # Vérification des tailles des ensembles
    print(f"Taille Train: {train_size}, Taille Validation: {val_size}")

    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in zip(X_train_dataloader, y_train_dataloader):
            # X_batch = tf.convert_to_tensor(X_batch, dtype=tf.complex64)
            # y_batch = tf.convert_to_tensor(y_batch, dtype=tf.complex64)
            # X_batch = tf.cast(X_batch, tf.complex64)
            # y_batch = tf.cast(y_batch, tf.complex64)

            with tf.GradientTape() as tape:
                predictions = model(X_batch)
                loss = model.loss(y_batch, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
        
        avg_train_loss = epoch_loss / (len(X_train_dataloader) / batch_size)
        train_losses.append(avg_train_loss)

        # Validation loop
        epoch_val_loss = 0
        for X_batch, y_batch in zip(X_val_dataloader, y_val_dataloader):
            # X_batch = tf.convert_to_tensor(X_batch, dtype=tf.complex64)
            # y_batch = tf.convert_to_tensor(y_batch, dtype=tf.complex64)
            # X_batch = tf.cast(X_batch, tf.complex64)
            # y_batch = tf.cast(y_batch, tf.complex64)

            predictions = model(X_batch)
            loss = model.loss(y_batch, predictions)
            # loss = model.loss(tf.complex(tf.math.real(y_batch), tf.math.imag(y_batch)), 
            #                   tf.complex(tf.math.real(predictions), tf.math.imag(predictions)))
            epoch_val_loss += loss.numpy()

        avg_val_loss = epoch_val_loss / (len(X_val_dataloader) / batch_size)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    end_time = time.time()  # Arrêter le chronomètre
    execution_time = end_time - start_time
    print(f"Temps d'exécution total: {execution_time:.2f} secondes")

    # Improved Plot for Training and Validation Loss
    plt.figure(figsize=(10, 6))  # Increase figure size for better readability
    # Plot Train Loss with smooth line and visible markers
    plt.plot(range(1, epochs+1), train_losses, linestyle='-', color='royalblue', linewidth=2, markersize=4, label="Train Loss")
    # Plot Validation Loss with dashed line and square markers
    plt.plot(range(1, epochs+1), val_losses, linestyle='--', color='orangered', linewidth=2, markersize=4, label="Validation Loss")
    # Labels and Title
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training & Validation Loss', fontsize=16, fontweight='bold')
    # Improved Legend
    plt.legend(fontsize=12, loc='upper right', frameon=True, edgecolor='black')
    # Enhanced Grid with Transparency
    plt.grid(True, linestyle='--', alpha=0.6)
    # Show the plot
    plt.show()

