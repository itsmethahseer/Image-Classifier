import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
CUDA_LAUNCH_BLOCKING=1
# Path to your dataset
data_dir = '/home/thahseer/Desktop/image_classification/data'

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to same size
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization values for pretrained models
                         std=[0.229, 0.224, 0.225])
])

# Dataset and Dataloader
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get class names
class_names = dataset.classes
num_classes = len(class_names)

# Load a pretrained model
model = models.resnet18(pretrained=True)

# Freeze earlier layers (optional)
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for your number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop (1 epoch for simplicity)
for epoch in range(30):  
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}], Loss: {running_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), 'model.pth')
print(f"Model saved with {num_classes} classes: {class_names}")