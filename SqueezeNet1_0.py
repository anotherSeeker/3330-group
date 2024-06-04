import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Enabling Cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {}'.format(device))
torch.device(device)

# Hyperparameters
num_epochs = 15
learning_rate = 0.0005
momentum = 0.9
weight_decay = 0.001

# Path to datasets
train_dataset_path = './datasets/FoodTrain1'
valid_dataset_path = './datasets/FoodValidate1'

# Transformations needed to make the data work with the model
squeeze_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Applies the transformations
train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=squeeze_transforms)
valid_dataset = torchvision.datasets.ImageFolder(root=valid_dataset_path, transform=squeeze_transforms)

# Data loader for dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False)

# Best loss and checkpoint function used for saving model
checkpoint_name = 'SqueezeNet'
def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'best accuracy': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, checkpoint_name+'.pth.tar')
    print(f'Saved model on epoch: {epoch+1} with accuracy: {best_acc:.2f}%')

def model_trainer():
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    epoch_label = []
    acc_label = []

    best_acc = 0.0

    # Training and validation loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
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

            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
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

        # Gathering data for plot
        epoch_label.append(epoch + 1)
        acc_label.append(accuracy)

        if best_acc < accuracy:
            best_acc = accuracy
            save_checkpoint(model, epoch, optimizer, best_acc)

    return epoch_label, acc_label

# Plot data to display accuracy graph
all_epoch_labels = []
all_acc_labels = []

# Loops over the training, so multiple model instances can be compared for a more general look at performance - if set to 1 loop, will only produce the model once
for i in range(1):
    # Model type: SqueezeNet1.0
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, 40, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = 40
    model = model.to(device)

    print(f'Training iteration {i+1}')
    epoch_label, acc_label = model_trainer()
    all_epoch_labels.append(epoch_label)
    all_acc_labels.append(acc_label)

# Plotting
plt.figure()
for i in range(1):
    plt.plot(all_epoch_labels[i], all_acc_labels[i], marker='o', linestyle='-', label=f'Test Model #{i+1}')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.title('Accuracy Progression')
plt.legend()
plt.show()