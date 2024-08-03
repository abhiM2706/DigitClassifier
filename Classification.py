import struct
import time
from array import array

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import numpy as np

import torch
from PIL import Image
import torchvision.transforms as transforms


import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Net import Net
from TransformImage import read_human_images


def read_images_labels(images_filepath, labels_filepath):
    labels = []
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", file.read())

    with open(images_filepath, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())

    images = np.array(image_data, dtype=np.float32).reshape(size, rows, cols)
    images /= 255.0  # Scale pixel values between 0 and 1

    return images, np.array(labels)


x_train, y_train = read_images_labels('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
x_test, y_test = read_images_labels('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')


# Convert data to tensors
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)



x_tensor_train = torch.tensor(x_train, dtype=torch.float32)
y_tensor_train = torch.tensor(y_train, dtype=torch.long)
x_tensor_test = torch.tensor(x_test, dtype=torch.float32)
y_tensor_test = torch.tensor(y_test, dtype=torch.long)
x_tensor_val = torch.tensor(x_val, dtype=torch.float32)
y_tensor_val = torch.tensor(y_val, dtype=torch.long)

x_tensor_train = x_tensor_train.view(54000, 1, 28, 28)
x_tensor_test = x_tensor_test.view(10000, 1, 28, 28)
x_tensor_val = x_tensor_val.view(6000, 1, 28, 28)

# Create datasets and dataloaders
train_dataset = TensorDataset(x_tensor_train, y_tensor_train)
test_dataset = TensorDataset(x_tensor_test, y_tensor_test)
val_dataset = TensorDataset(x_tensor_val, y_tensor_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# Define the neural network, criterion, and optimizer
net = Net()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for i, train_data in enumerate(train_loader):
        optimizer.zero_grad()

        # dataloader_train data is in a tuple of inputs and labels of flowers
        inputs, labels = train_data

        # Make predictions for this batch
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        accuracy = (predicted == labels).float().sum().item() / labels.size(dim=0)
        running_loss += loss.item()

    last_loss = running_loss / len(train_loader)
    print('  Epoch {} loss: {}'.format(epoch_index, last_loss))

    return last_loss

num_epochs = 50
best_test_loss = 1_000_000.


for epoch in tqdm(range(num_epochs)):
    net.train()
    avg_loss = train(num_epochs)
    net.eval()
    val_running_loss = 0.0
    best_val_loss=10000
    model_path = 'best_model.pt'

    for i, data in enumerate(val_loader):
        val_input, val_labels = data
        outputs = net(val_input)

        vloss = loss_fn(outputs, val_labels)
        val_running_loss += vloss

    vloss = val_running_loss / len(val_loader)
    if (vloss < best_val_loss):
        torch.save(net.state_dict(), model_path)
        best_val_loss = vloss


net.eval()
test_running_loss = 0.0
correct = 0
total=0
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        inputs, labels = test_data
        outputs = net(inputs)
        test_loss = loss_fn(outputs, labels)
        test_running_loss += test_loss
        _, predicted_classes = torch.max(outputs, 1)
        accuracy = (predicted_classes == labels).float().sum().item() / labels.size(dim=0)
        correct += accuracy
        total+=1

avg_test_loss = test_running_loss / (i + 1)
print('LOSS train {} valid {}'.format(avg_loss, avg_test_loss))

num_epochs += 1
print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')