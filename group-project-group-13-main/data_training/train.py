import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np

# Enhanced transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=1.5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Reload and re-split datasets with the new transformations
full_dataset = datasets.ImageFolder('Planets and Moons', transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load a pretrained ResNet18 model and modify the last fully connected layer
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features  # Get the number of inputs for the last layer
model.fc = torch.nn.Linear(num_features, 4)  # Adjust for 4 classes: Earth, Mars, Moon, Mercury

# Move the model to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to train the model
def train_model(model, criterion, optimizer, train_loader, epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Clear gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the weights
            total_loss += loss.item()  # Accumulate the loss

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    # Save the model state
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to compute gradients
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get predictions from the maximum value
            total += labels.size(0)  # Total number of labels
            correct += (predicted == labels).sum().item()  # Total correct predictions

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')

# Call the training and evaluation functions
train_model(model, criterion, optimizer, train_loader)
evaluate_model(model, test_loader)
