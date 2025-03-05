from data_gen_func import data_gen
from MLP import MLP

model = MLP()
print(model)

input_train, output_train = data_gen()
input_val, output_val = data_gen(size=100)

