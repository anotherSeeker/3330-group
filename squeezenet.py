import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# Enabling Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))
torch.device(device)

# Hyperparameters
num_epochs = 15
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0005

# Path to datasets
train_dataset_path = './datasets/FoodTrain1'
valid_dataset_path = './datasets/FoodValidate1'

# Model type: SqueezeNet1.1 https://arxiv.org/abs/1602.07360
model = models.squeezenet1_0(pretrained=True)
model.eval()

# SqueezeNet1.1 has 1000 classes by default, adding final layer for 40 classes
num_classes = 40  
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
model.num_classes = num_classes

model = model.to(device)

# Transformations needed to make the data work with the model
squeeze_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Applies the trainformations
train_dataset = torchvision.datasets.ImageFolder(root = train_dataset_path, transform = squeeze_transforms)
valid_dataset = torchvision.datasets.ImageFolder(root = valid_dataset_path, transform = squeeze_transforms)

# Data loader for dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 32, shuffle = False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# Training and validation loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')



