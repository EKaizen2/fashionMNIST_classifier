

# fashionMNIST_classifier GitHub Repository: https://github.com/EKaizen2/fashionMNIST_classifier.git 

Makefile usage:
make or make all: Sets up the project and runs the classifier
make setup: Creates necessary directories and downloads the dataset
make run: Runs the classifier
make clean: Removes all generated files (logs, cache, dataset)
make clean-logs: Removes only the log files
make logs: Displays the contents of logs.txt
make deps: Installs required Python packages
make help: Shows help information about available commands


# Fashion MNIST Neural Network Classifier

## Overview
This project implements an Artificial Neural Network (ANN) classifier for the Fashion MNIST dataset. The classifier can identify 10 different types of fashion items from grayscale images, including clothing, accessories, and footwear.

## Requirements
- Python 3.7+
- PyTorch
- torchvision
- PIL (Pillow)
- numpy
- matplotlib

## Installation

## Usage
1. Run the classifier:
```bash
    python classifier.py
```

2. The model will train and display progress
3. After training, you can input paths to fashion item images (.jpg or .png)
4. Type 'exit' to quit

## Dataset
The Fashion MNIST dataset consists of:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 classes of fashion items

## Model Architecture
- Input layer: 784 neurons (28x28 flattened)
- Hidden layers: 512 → 256 → 128 neurons
- Output layer: 10 neurons (classes)
- ReLU activation and dropout for regularization


