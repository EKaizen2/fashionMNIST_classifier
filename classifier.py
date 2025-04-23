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
import datetime
import os


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


# Define the Neural Network
class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_file = 'logs/logs.txt'
    
    # Add session separator if file exists
    if os.path.exists(log_file):
        with open(log_file, 'a') as f:
            f.write('\n' + '='*50 + '\n')
    
    return log_file

def log_message(log_file, message, file_handle=None):
    print(message)  # Still print to console
    if file_handle:
        file_handle.write(message + '\n')
        file_handle.flush()  # Ensure the message is written immediately

def train_model(epochs=5, log_file=None, file_handle=None):
    model.train()
    log_message(log_file, f"\nStarting Training at {datetime.datetime.now()}", file_handle)
    log_message(log_file, f"Training on device: {device}", file_handle)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                log_msg = f'Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {running_loss / 100:.4f}'
                log_message(log_file, log_msg, file_handle)
                running_loss = 0.0
        
        log_message(log_file, f'Completed Epoch {epoch + 1}', file_handle)

def test_model(log_file=None, file_handle=None):
    model.eval()
    correct = 0
    total = 0
    
    log_message(log_file, "\nStarting Testing Phase", file_handle)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    log_message(log_file, f'Test Accuracy: {accuracy:.2f}%', file_handle)
    return accuracy

def predict_image(image_path, log_file=None, file_handle=None):
    try:
        # Load image using PIL
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            class_name = train_dataset.classes[predicted.item()]
            
            log_message(log_file, f"Prediction for {image_path}: {class_name}", file_handle)
            return class_name
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        log_message(log_file, error_msg, file_handle)
        return error_msg

# Main execution
if __name__ == "__main__":
    # Setup logging
    log_file = setup_logger()
    
    # Open log file for the session
    with open(log_file, 'a') as log_handle:
        # Log initial information
        log_message(log_file, "Fashion MNIST Classifier Training Session", log_handle)
        log_message(log_file, f"Start Time: {datetime.datetime.now()}", log_handle)
        log_message(log_file, f"Device: {device}", log_handle)
        log_message(log_file, "\nModel Architecture:", log_handle)
        log_message(log_file, str(model), log_handle)
        
        # Log dataset information
        log_message(log_file, f"\nTraining Set Size: {len(train_dataset)}", log_handle)
        log_message(log_file, f"Test Set Size: {len(test_dataset)}", log_handle)
        log_message(log_file, f"Classes: {train_dataset.classes}", log_handle)
        
        print("Training the model...")
        train_model(epochs=5, log_file=log_file, file_handle=log_handle)
        
        print("\nTesting the model...")
        test_accuracy = test_model(log_file=log_file, file_handle=log_handle)
        
        print("\nModel ready for predictions!")
        while True:
            filepath = input("\nPlease enter a filepath (or 'exit' to quit): ")
            if filepath.lower() == 'exit':
                log_message(log_file, "\nSession ended by user", log_handle)
                print("Exiting...")
                break
            
            prediction = predict_image(filepath, log_file=log_file, file_handle=log_handle)
            print(f"Classifier: {prediction}")


