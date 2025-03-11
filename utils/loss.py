import torch.nn as nn

def complex_mse_loss(output, target):
    """
    Calcule la perte MSE pour les tenseurs complexes.
    """
    real_loss = nn.functional.mse_loss(output.real, target.real)
    imag_loss = nn.functional.mse_loss(output.imag, target.imag)
    return real_loss + imag_loss