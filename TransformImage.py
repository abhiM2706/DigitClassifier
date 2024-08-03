import struct
import time
from array import array

import PIL
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import numpy as np

import torch
from PIL import Image

import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt

from Net import Net

def read_human_images(image):
    transform = transforms.Compose([transforms.Resize(28), transforms.CenterCrop(28)
                                       , transforms.Grayscale(1)])
    img_tensor = transform(image)
    return img_tensor

def predict(image_path):
    image = PIL.Image.open(image_path)
    img_tensor = read_human_images(image)
    img_np_array = np.array(img_tensor, dtype=np.float32)

    img_np_array /= 255
    img_np_array = 1-img_np_array
    plt.imshow(img_np_array)
    plt.savefig('plot.png')

    img_np_array = torch.tensor(img_np_array, dtype=torch.float32)
    print(img_np_array)
    img_np_array = img_np_array.view(1, 1, 28, 28)
    net = Net()
    checkpoint = torch.load('best_model.pt')
    net.load_state_dict(checkpoint)
    net.eval()
    outputs = net(img_np_array)

    _, predicted_classes = torch.max(outputs, 1)

    return predicted_classes
