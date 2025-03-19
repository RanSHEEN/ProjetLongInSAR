import torch.nn as nn
import torch

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
