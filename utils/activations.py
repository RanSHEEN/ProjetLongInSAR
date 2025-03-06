import numpy as np
import torch

def zReLU(z):
    """
    zReLU laisse passer les entrées situées dans le premier quadrant du domaine complexe.
    Input :
        - z : un tenseur complexe PyTorch de type torch.complex64.
    Output :
        - res : un tenseur complexe PyTorch de même forme que z, avec zReLU appliqué.
    """
    # Calculer l'angle (phase) de chaque nombre complexe
    angle = torch.angle(z)  # angle en radians

    # Créer un masque pour les nombres complexes dans le premier quadrant
    mask = ((angle > 0) & (angle < np.pi / 2)).float()

    # Appliquer le masque à z
    res = z * mask

    return res

def cReLU(z):
    """
    
    """