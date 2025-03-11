import torch
from utils.loss import weighted_arg_mod_loss

x = torch.randn(2, dtype=torch.cfloat)
print(x)
print(0.5 * torch.abs(torch.angle(x[0]) - torch.angle(x[1]))**2 + (1 - 0.5) * torch.abs(torch.abs(x[0]) - torch.abs(x[1]))**2)
print(weighted_arg_mod_loss(x[0], x[1]))