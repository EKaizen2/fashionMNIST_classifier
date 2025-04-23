import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image


# Data loading and preprocessing
DATA_DIR = "."
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)
test_dataset = datasets.FashionMNIST(DATA_DIR, train=False, download=False, transform=transform)

print(train_dataset)
print(test_dataset)
print(train_dataset.classes)

print(train_dataset.data.shape)
print(test_dataset.data.shape)

print(train_dataset.targets.shape)
print(test_dataset.targets.shape)

print(train_dataset.targets)
print(test_dataset.targets)