import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os
from torch.amp import GradScaler, autocast
from tqdm import tqdm  # Progress bar library

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
base_path = "C:\\Employee Welbeing through Emotion Detection\\The Solution"
train_path = f"{base_path}\\grok\\data\\train"
test_path = f"{base_path}\\grok\\data\\test"
save_dir = f"{base_path}\\saved_emotion_model"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}\\Assets", exist_ok=True)
os.makedirs(f"{save_dir}\\Variables", exist_ok=True)

# Data transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load datasets
print("Loading train dataset...")
train_dataset = ImageFolder(train_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)
print(f"Train dataset loaded. Classes: {train_dataset.classes}, Images: {len(train_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
print("Data loaders created.")

# Custom CNN
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function with progress bars
def train_model(model, train_loader, criterion, optimizer, epochs, name):
    scaler = GradScaler('cuda')
    losses, accuracies = [], []
    
    # Outer progress bar for epochs
    with tqdm(total=epochs, desc=f"Training {name}", unit="epoch") as epoch_pbar:
        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0, 0, 0
            
            # Inner progress bar for batches
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False)
            for inputs, labels in batch_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # Update batch progress bar with current loss
                batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            losses.append(running_loss / len(train_loader))
            accuracies.append(correct / total)
            # Update epoch progress bar with metrics
            epoch_pbar.set_postfix({'loss': f"{losses[-1]:.4f}", 'acc': f"{accuracies[-1]:.4f}"})
            epoch_pbar.update(1)
    
    return losses, accuracies

# Initialize models
resnet = models.resnet50(weights="IMAGENET1K_V1")
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 7)
resnet = resnet.to(device)

cnn = CustomCNN().to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer_resnet = torch.optim.Adam(resnet.parameters(), lr=0.001)
optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=0.001)

# Train models
print("Starting ResNet50 training...")
resnet_losses, resnet_accs = train_model(resnet, train_loader, criterion, optimizer_resnet, 25, "ResNet50")
print("Starting CustomCNN training...")
cnn_losses, cnn_accs = train_model(cnn, train_loader, criterion, optimizer_cnn, 25, "CustomCNN")

# Plot loss and accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(resnet_losses, label="ResNet50")
plt.plot(cnn_losses, label="CustomCNN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(resnet_accs, label="ResNet50")
plt.plot(cnn_accs, label="CustomCNN")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.savefig(f"{save_dir}\\Assets\\training_plots.png")
plt.close()

# Save models
torch.save(resnet.state_dict(), f"{save_dir}\\Variables\\resnet_best.pth")
torch.save(cnn.state_dict(), f"{save_dir}\\Variables\\cnn.pth")
print("Training complete. Models saved.")