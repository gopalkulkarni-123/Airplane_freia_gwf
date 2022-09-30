# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
#from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.annotations = pd.read_csv(r"E:\Masters\Univerities\TU Dresden\Post_Admit\Studies\4th Sem\RP\Airplane\airplane_data_v8.csv")
        #self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        values_path = self.annotations.iloc[index, 0]
        points = values_path
        labels = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            points = self.transform(points)

        return (points,labels)
    
    # Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 32
num_epochs = 10

# Load Data
dataset = CustomDataset()

# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.
#train_set, test_set = torch.utils.data.random_split(dataset, [5, 5])
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data
        targets = targets

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")