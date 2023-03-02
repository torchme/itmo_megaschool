import os
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioDataset, AudioTransform
from model import UNet

train_path = os.path.join(sys.path[0], "./train_dataset/")
valid_path = os.path.join(sys.path[0], "./valid_dataset/")

num_epochs = 10
batch_size = 32
learning_rate = 0.001


try:
    transform = AudioTransform(probability=0.8)
    print("Transform")
except:
    transform = None

if os.path.exists(train_path):
    train_dataset = AudioDataset(train_path, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = AudioDataset(valid_path, transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
else:
    FileExistsError()


model = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loss_list = []
val_loss_list = []
train_mse_list = []
val_mse_list = []

for epoch in range(num_epochs):
    epoch_train_loss = 0
    epoch_val_loss = 0
    epoch_train_mse = 0
    epoch_val_mse = 0

    # Training loop
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{num_epochs}"):
        noisy_mel, clean_mel = batch
        noisy_mel = noisy_mel.to(device)
        clean_mel = clean_mel.to(device)

        noisy_mel = noisy_mel.unsqueeze(1)  # Add channel dimension
        clean_mel = clean_mel.unsqueeze(1)  # Add channel dimension

        optimizer.zero_grad()

        reconstructed_mel = model(noisy_mel)
        print(noisy_mel.shape, clean_mel.shape, reconstructed_mel.shape)

        loss = criterion(reconstructed_mel, clean_mel)
        print(f"Train loss: {loss}")
        loss.backward()
        optimizer.step()

        ## Compute MSE loss
        # mse = mean_squared_error(reconstructed_mel.cpu().detach().numpy(),
        #                         clean_mel.cpu().detach().numpy())
        # epoch_train_mse += mse
        # epoch_train_loss += loss.item()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc=f"Val Epoch {epoch+1}/{num_epochs}"):
            noisy_mel, clean_mel = batch
            noisy_mel = noisy_mel.to(device)
            clean_mel = clean_mel.to(device)

            noisy_mel = noisy_mel.unsqueeze(1)  # Add channel dimension
            clean_mel = clean_mel.unsqueeze(1)  # Add channel dimension

            reconstructed_mel = model(noisy_mel)

            loss = criterion(reconstructed_mel, clean_mel)
            print(f"Valid loss: {loss}")

            ## Compute MSE loss
            # mse = mean_squared_error(reconstructed_mel.cpu().detach().numpy(),
            #                         clean_mel.cpu().detach().numpy())
            # epoch_val_mse += mse
            # epoch_val_loss += loss.item()

    ## Add loss and MSE to lists for plotting
    # train_loss_list.append(epoch_train_loss)
    ## Print losses
    # epoch_train_loss /= len(train_dataloader)
    # epoch_val_loss /= len(valid_dataloader)
    # print(f'Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}')

    ## Save best model
    # if epoch_val_loss < best_val_loss:
    #    torch.save(model.state_dict(), 'best_model.pt')
    #    best_val_loss = epoch_val_loss
