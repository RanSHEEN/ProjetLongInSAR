import torch.nn as nn
import torch
import numpy as np

def complex_mse_loss(output, target):
    """
    Calcule la perte MSE pour les tenseurs complexes.
    """
    real_loss = nn.functional.mse_loss(output.real, target.real)
    imag_loss = nn.functional.mse_loss(output.imag, target.imag)
    return real_loss + imag_loss

def weighted_arg_mod_loss(output, target, w=0.5):
    """
    Calcule w * |output_theta - target_theta| + (1 - w) * |output_mod - target_mod|
    """
    return w * nn.functional.mse_loss(torch.angle(output), torch.angle(target)) + (1 - w) * nn.functional.mse_loss(torch.abs(output), torch.abs(target))


def euclidean_distance_complex(u, v):
    # Calculer la magnitude des vecteurs complexes
    u_magnitude = np.abs(u)
    v_magnitude = np.abs(v)

    # Calculer la distance euclidienne
    distance = np.sqrt(np.sum((u_magnitude - v_magnitude) ** 2))
    return distance

def complex_mse_loss_np(output, target):
    """
    Calcule la perte MSE pour les tableaux NumPy complexes.
    """
    # Convertir les entrées en tableaux NumPy si nécessaire
    output = np.array(output)
    target = np.array(target)

    # Calculer la perte MSE pour les parties réelles et imaginaires
    real_loss = np.mean((output.real - target.real) ** 2)
    imag_loss = np.mean((output.imag - target.imag) ** 2)

    return real_loss + imag_loss


def euclidean_distance_complex_list(output, target):
    """
    Calcule une liste de distances euclidiennes pour les tableaux NumPy complexes.
    Chaque distance est calculée entre les vecteurs correspondants de output et target.

    :param output: Tableau NumPy de forme (n, d)
    :param target: Tableau NumPy de forme (n, d)
    :return: Liste de distances euclidiennes
    """
    # Calculer la distance euclidienne pour chaque paire de vecteurs
    distances = [np.sqrt(np.sum(np.abs(output[i] - target[i]) ** 2)) for i in range(len(output))]
    return distances