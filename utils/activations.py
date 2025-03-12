import numpy as np
import torch
import torch.nn as nn

class CartReLU(nn.Module):
    """
    Applique la fonction d'activation cartReLU à un tenseur complexe.

    La cartReLU applique une ReLU séparément sur les parties réelle et imaginaire du tenseur complexe z. Cela revient à appliquer une ReLU indépendante sur chaque composante.

    Args:
        z (torch.Tensor): Un tenseur complexe de type torch.complex64.

    Returns:
        torch.Tensor: Un tenseur complexe de même forme que z, après application de la cartReLU.
    """
    def __init__(self):
        super(CartReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, z):
        a, b = z.real, z.imag
        a_relu = nn.ReLU(a)
        b_relu = nn.ReLU(b)
        res = torch.complex(a_relu, b_relu)
        return res


class ZReLU(nn.Module):
    """
    Applique la fonction d'activation zReLU à un tenseur complexe.

    La zReLU conserve uniquement les entrées situées dans le premier quadrant du domaine complexe (c'est-à-dire les nombres complexes dont la phase est comprise entre 0 et π/2). Les autres entrées sont annulées.

    Args:
        z (torch.Tensor): Un tenseur complexe de type torch.complex64.

    Returns:
        torch.Tensor: Un tenseur complexe de même forme que z, après application de la zReLU. Les entrées en dehors du premier quadrant sont mises à zéro.
    """
    def forward(self, z):
        # Calculer l'angle (phase) de chaque nombre complexe
        angle = torch.angle(z)  # angle en radians

        # Créer un masque pour les nombres complexes dans le premier quadrant
        mask = ((angle > 0) & (angle < np.pi / 2)).float()

        # Appliquer le masque à z
        res = z * mask

        return res


class Cardioid(nn.Module):
    """
    Applique la fonction d'activation Cardioid à un tenseur complexe.

    La Cardioid préserve la phase du nombre complexe tout en modulant sa magnitude en fonction de cette phase :
    - Pour les phases proches de 0, la magnitude de z est peu atténuée.
    - Pour les phases proches de π ou -π, la magnitude de z est fortement atténuée.
    - Pour les phases intermédiaires, la magnitude est atténuée de manière proportionnelle à cos(θ).

    Args:
        z (torch.Tensor): Un tenseur complexe de type torch.complex64.

    Returns:
        torch.Tensor: Un tenseur complexe de même forme que z, après application de la Cardioid.
    """
    def forward(self, z):
        angle = torch.angle(z)
        attenuation_factor = (1 + torch.cos(angle)) / 2
        return attenuation_factor * z


# class ModReLU(nn.Module):
#     """
#     Applique la fonction d'activation modReLU à un tenseur complexe.

#     La modReLU agit uniquement sur le module (magnitude) de z tout en conservant sa phase. Elle introduit une zone morte autour de 0, annulant les vecteurs dont la magnitude est inférieure à un biais b. Ce biais est un paramètre apprenable.

#     Args:
#         z (torch.Tensor): Un tenseur complexe de type torch.complex64.

#     Returns:
#         torch.Tensor: Un tenseur complexe de même forme que z, après application de la modReLU.
#     """
#     def __init__(self, in_features):
#         super(ModReLU, self).__init__()
#         self.log_neg_b = nn.Parameter(torch.zeros(in_features))

#     def forward(self, z):
#         b = -torch.exp(self.log_neg_b)
#         b = b.unsqueeze(0).expand(z.size(0), -1)
#         r = torch.abs(z)
#         output = torch.relu(r + b)
#         output = output * (z / (r + 1e-8))
#         return output

class ModReLU(nn.Module):
    """
    Applique la fonction d'activation modReLU à un tenseur complexe.

    La modReLU agit uniquement sur le module (magnitude) de z tout en conservant sa phase. Elle introduit une zone morte autour de 0, annulant les vecteurs dont la magnitude est inférieure à un biais b. Ce biais est un paramètre apprenable.

    Args:
        z (torch.Tensor): Un tenseur complexe de type torch.complex64.

    Returns:
        torch.Tensor: Un tenseur complexe de même forme que z, après application de la modReLU.
    """
    def __init__(self, in_features):
        super(ModReLU, self).__init__()
        self.b = -0.1  # Définir b comme une constante

    def forward(self, z):
        b = self.b
        r = torch.abs(z)
        output = torch.relu(r + b)
        output = output * (z / (r + 1e-8))
        return output

class AffSin(nn.Module):
    """
    Applique AffSin : z -> f(Re(z)) + i * f(Im(z))

    avec f : x -> x + sin(pi * x)

    Args:
        z (torch.Tensor): Un tenseur complexe de type torch.complex64.

    Returns:
        torch.Tensor: Un tenseur complexe de même forme que z, après application de AffSin.
    """
    def __init__(self, alpha=1):
        super(AffSin, self).__init__()
        self.alpha = alpha
    def forward(self, z):
        return torch.complex((z.real + self.alpha * torch.sin(torch.pi * z.real)), (z.imag + self.alpha * torch.sin(torch.pi * z.imag)))
    
class MVN(nn.Module):
    """
    Applique MVN à un tenseur complexe, MVN correspond une division de R² en K cadrants.

    Au sein d'un même cadrant tous les points sont projetés sur le même point appartenant au cercle unitaire.

    f: z -> exp(i * 2 * pi * a / k) 
    
    avec i * 2 * pi * a < arg(z) < i * 2 * pi * (a + 1)

    Args:
        z (torch.Tensor): Un tenseur complexe de type torch.complex64.

    Returns:
        torch.Tensor: Un tenseur complexe de même forme que z, après application de MVN.
    """