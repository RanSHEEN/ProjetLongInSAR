import torch
from utils.loss import weighted_arg_mod_loss
from utils.data_gen_func import data_gen

X, w = data_gen(size=1, p=10, n=25)
print(w)
