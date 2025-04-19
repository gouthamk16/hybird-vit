import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Configuration
data_dir = '/content/training_data/train'        
num_classes = 101        
batch_size = 32            
num_epochs = 10            
learning_rate = 1e-4       
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations and preprocessing
# Using standard normalization for pretrained models
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Model: Fine-tune a pretrained ResNet50
model = models.resnet50(pretrained=True)
# Replace the final fully-connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')

# Save the trained model
os.makedirs('checkpoints', exist_ok=True)
checkpoint_path = os.path.join('checkpoints', 'resnet50_cats101.pth')
torch.save(model.state_dict(), checkpoint_path)
print(f'Model saved to {checkpoint_path}')


## Validation - comment out during train
model.eval()
val_loss = 0.0
val_correct = 0
val_total = 0

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)

val_loss /= val_total
val_acc = val_correct / val_total
print(f'Validation -> Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')