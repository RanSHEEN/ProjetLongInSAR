import torch
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tensorflow as tf
from torch.utils.data import DataLoader

from cvnn.losses import ComplexMeanSquareError

import data_gen
import neural_network

# @tf.function
# def train_step(X_batch, y_batch):
#     with tf.GradientTape() as tape:
#         predictions = model(X_batch)
#         loss = model.loss(y_batch, predictions)

#     gradients = tape.gradient(loss, model.trainable_variables)
#     model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return loss

if __name__ == "__main__":
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
    

    
    # Convertir les listes en tableaux NumPy uniques
    X_list = np.array(X_list)  # Convertir en un seul tableau NumPy
    w_theta_list = np.array(w_theta_list)      # Convertir en un seul tableau NumPy

    # Convertir en tenseurs complexes
    X_list_tensor = torch.tensor(X_list, dtype=torch.complex64)
    w_theta_list_tensor = torch.tensor(w_theta_list, dtype=torch.complex64)

    # Passage au format ComplexDataset
    X_train = data_gen.ComplexDataset(X_list_tensor)
    y_train = data_gen.ComplexDataset(w_theta_list_tensor)

    n = 64 # number of samples
    # input_dim = p * n + p * p  # X est de taille (p, n) et Sigma_tilde de taille (p, p)
    input_dim = n
    hidden_dim = 128  # Nombre de neurones cachés
    output_dim = p  # Sortie (ex. estimation de covariance)

    model = neural_network.NeuralNetwork(input_dim, hidden_dim, output_dim) 
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss=ComplexMeanSquareError())
    for layer in model.layers:
        print(f"Layer {layer.name} trainable variables:", layer.trainable_variables)

    epochs = 10
    batch_size = 32
    losses = []
    X_train_dataloader = DataLoader(X_train, batch_size=batch_size)
    y_train_dataloader = DataLoader(y_train, batch_size=batch_size)
    for epoch in range(epochs):
        epoch_loss = 0
        # for i in range(0, len(X_list_tensor), batch_size):
        for X_batch, y_batch in zip(X_train_dataloader, y_train_dataloader):
            X_batch = tf.convert_to_tensor(X_batch, dtype=tf.complex64)
            y_batch = tf.convert_to_tensor(y_batch, dtype=tf.complex64)
            # X_batch = tf.cast(X_batch, tf.complex64)
            # y_batch = tf.cast(y_batch, tf.complex64)

            # X_batch = tf.convert_to_tensor(X_train_dataloader[i:i+batch_size], dtype=tf.complex64)
            # # Sigma_tilde_batch = tf.convert_to_tensor(Sigma_tilde_list[i:i+batch_size], dtype=tf.complex64)
            # y_batch = tf.convert_to_tensor(y_train_dataloader[i:i+batch_size], dtype=tf.complex64)

            with tf.GradientTape() as tape:
                predictions = model(X_batch)
                loss = model.loss(y_batch, predictions)

            # loss = train_step(X_batch, y_batch)
            gradients = tape.gradient(loss, model.trainable_variables)
            print("Gradients:", gradients)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss += loss.numpy()
        
        avg_loss = epoch_loss / (len(X_train_dataloader) / batch_size)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Plot loss
    plt.plot(range(1, epochs+1), losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid()
    plt.show()

