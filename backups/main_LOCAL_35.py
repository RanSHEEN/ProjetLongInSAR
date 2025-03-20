import torch
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from torch.utils.data import DataLoader

from cvnn.losses import ComplexMeanSquareError
import cvnn.layers as complex_layers
from cvnn.activations import zrelu, modrelu

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
    
    # Normalize X_list
    real_parts = np.array([np.real(x) for x in X_list])
    imag_parts = np.array([np.imag(x) for x in X_list])
    
    # Normalize real and imaginary parts separately
    real_mean, real_std = np.mean(real_parts), np.std(real_parts)
    imag_mean, imag_std = np.mean(imag_parts), np.std(imag_parts)
    
    normalized_real = (real_parts - real_mean) / real_std
    normalized_imag = (imag_parts - imag_mean) / imag_std
    
    # Reconstruct complex values
    X_list = normalized_real + 1j * normalized_imag

    # Normalize Sigma_tilde_list
    Sigma_tilde_list = [cov / np.linalg.norm(cov, 'fro') for cov in Sigma_tilde_list]

    # Normalize w_theta_list
    w_theta_list = [w / np.linalg.norm(w) if np.linalg.norm(w) != 0 else w for w in w_theta_list]

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
    epochs = 300
    batch_size = 32
    train_losses = []
    val_losses = []
    X_train_dataloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    y_train_dataloader = DataLoader(y_train, batch_size=batch_size, shuffle=True)
    X_val_dataloader = DataLoader(X_val, batch_size=batch_size)
    y_val_dataloader = DataLoader(y_val, batch_size=batch_size)

    # Vérification des tailles des ensembles
    print(f"Taille Train: {train_size}, Taille Validation: {val_size}")

    # # Early Stopping Parameters
    # patience = 20  # Number of epochs to wait before stopping
    # min_delta = 1e-4  # Minimum improvement to be considered as an improvement
    # best_val_loss = float('inf')
    # epochs_without_improvement = 200

    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in zip(X_train_dataloader, y_train_dataloader):
            #%% SGD partie réelle
            with tf.GradientTape(persistent=True) as tape:
                predictions = model(X_batch)
                loss = model.loss.call(y_batch, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)

            # Ensure gradients remain complex
            # complex_gradients = [tf.complex(tf.math.real(g), tf.math.imag(g)) if g is not None else None for g in gradients]

            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
            
            #%% SGD partie réelle + complexe
            # with tf.GradientTape(persistent=True) as tape:
            #     predictions = model(X_batch)
            #     loss = model.loss.call(y_batch, predictions)

            # # Extract trainable variables
            # trainable_vars = model.trainable_variables

            # # Compute gradients for real and imaginary parts separately
            # real_gradients = []
            # imag_gradients = []

            # for var in trainable_vars:
            #     real_grad = tape.gradient(loss, var)  # Compute gradient w.r.t the complex variable

            #     # Ensure gradients are not None
            #     if real_grad is None:
            #         real_grad = tf.zeros_like(var)

            #     # Extract real and imaginary parts of the gradients
            #     real_grad_part = tf.math.real(real_grad)
            #     imag_grad_part = tf.math.imag(real_grad)

            #     real_gradients.append(real_grad_part)
            #     imag_gradients.append(imag_grad_part)

            # # Apply gradients separately for real and imaginary parts
            # for var, real_grad, imag_grad in zip(trainable_vars, real_gradients, imag_gradients):
            #     # Extract real and imaginary parts of the variable
            #     real_var = tf.Variable(tf.math.real(var))
            #     imag_var = tf.Variable(tf.math.imag(var))

            #     # Apply gradient descent update
            #     real_update = real_var - model.optimizer.learning_rate * real_grad
            #     imag_update = imag_var - model.optimizer.learning_rate * imag_grad

            #     # Merge back into a complex variable
            #     updated_var = tf.complex(real_update, imag_update)

            #     # Assign updated variable back to the model
            #     var.assign(updated_var)

            # # Delete the tape to free memory
            # del tape

        # avg_train_loss = epoch_loss / (len(X_train_dataloader) / batch_size)
        avg_train_loss = epoch_loss / len(X_train_dataloader)
        avg_train_loss = np.mean(avg_train_loss)  # Ensure it returns a scalar
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

        # avg_val_loss = epoch_val_loss / (len(X_val_dataloader) / batch_size)
        avg_val_loss = epoch_val_loss / len(X_val_dataloader)
        avg_val_loss = np.mean(avg_val_loss)
        val_losses.append(avg_val_loss)

        # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss.item():.4f}, Validation Loss: {avg_val_loss.item():.4f}")

        # # Early Stopping Check
        # if avg_val_loss < best_val_loss - min_delta:
        #     best_val_loss = avg_val_loss
        #     epochs_without_improvement = 0  # Reset counter
        # else:
        #     epochs_without_improvement += 1

        # if epochs_without_improvement >= patience:
        #     print(f"Early stopping triggered after {epoch+1} epochs.")
        #     break  # Stop training

    end_time = time.time()  # Arrêter le chronomètre
    execution_time = end_time - start_time
    print(f"Temps d'exécution total: {execution_time:.2f} secondes")

    # Plot Training & Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, linestyle='-', color='royalblue', linewidth=2, markersize=4, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, linestyle='--', color='orangered', linewidth=2, markersize=4, label="Validation Loss")
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training & Validation Loss', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right', frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    ## keras version
    # model = tf.keras.models.Sequential()
    # model.add(complex_layers.ComplexInput(input_shape=input_dim))
    # model.add(complex_layers.ComplexFlatten())
    # model.add(complex_layers.ComplexDense(units=64, activation='zrelu'))
    # model.add(complex_layers.ComplexDense(units=10, activation='modrelu'))
    # model.compile(loss=ComplexMeanSquareError(), optimizer='SGD', metrics=['accuracy', ComplexMeanSquareError()])

    # model.fit(X_train_dataloader, y_train_dataloader, epochs=epochs, batch_size=batch_size, validation_data=(X_val_dataloader, y_val_dataloader))

    # plt.plot(model.history.history['loss'], label='train')
    # plt.plot(model.history.history['val_loss'], label='validation')
    # plt.legend()
    # plt.show()