from utils.data_gen_func import data_gen
from utils.MLP import MLP, ComplexDataset
from utils.activations import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from utils.loss import complex_mse_loss
import torch.optim as optim
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = MLP(hidden_sizes=[1024, 512, 256, 128], activation='cardioid')
print(model)

input_train, output_train = data_gen()
input_val, output_val = data_gen(size=100)

# Convertir les listes en tableaux NumPy uniques
input_train = np.array(input_train)  # Convertir en un seul tableau NumPy
input_val = np.array(input_val)      # Convertir en un seul tableau NumPy
output_train = np.array(output_train)  # Convertir en un seul tableau NumPy
output_val = np.array(output_val)      # Convertir en un seul tableau NumPy

# Convertir en tenseurs complexes
input_train_tensor = torch.tensor(input_train, dtype=torch.complex64)
input_val_tensor = torch.tensor(input_val, dtype=torch.complex64)
output_train_tensor = torch.tensor(output_train, dtype=torch.complex64)
output_val_tensor = torch.tensor(output_val, dtype=torch.complex64)

# Passage au format ComplexDataset
in_train_dataset = ComplexDataset(input_train_tensor)
out_train_dataset = ComplexDataset(output_train_tensor)
in_val_dataset = ComplexDataset(input_val_tensor)
out_val_dataset = ComplexDataset(output_val_tensor)

# Création des dataloader
batch_size = 32
in_train_dataloader = DataLoader(in_train_dataset, batch_size=batch_size)
out_train_dataloader = DataLoader(out_train_dataset, batch_size=batch_size)
in_val_dataloader = DataLoader(in_val_dataset, batch_size=batch_size)
out_val_dataloader = DataLoader(out_val_dataset, batch_size=batch_size)

optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

# Boucle d'entraînement
num_epochs = 200
for epoch in range(num_epochs):
    model.train()  # Passer le modèle en mode entraînement
    train_loss = 0.0

    # Itérer sur les données d'entraînement
    for inputs, targets in zip(in_train_dataloader, out_train_dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = complex_mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(in_train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()  # Passer le modèle en mode évaluation
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in zip(in_val_dataloader, out_val_dataloader):
            outputs = model(inputs)
            loss = complex_mse_loss(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(in_val_dataloader)
    val_losses.append(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Tracer les pertes
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
