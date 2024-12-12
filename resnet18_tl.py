import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import datasets, models, transforms
from tqdm import tqdm

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"using device: {device}")

# Step 1: Data Preparation
transform = transforms.Compose(
    [
        transforms.Grayscale(
            num_output_channels=3
        ),  # Convert MNIST (1 channel) to 3 channels
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Model Setup
# Load pretrained ResNet18
weight = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weight)
model.fc = nn.Linear(model.fc.in_features, 10)  # Modify the final layer for 10 classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Step 3: Training the Model
epochs = 5
train_losses = []
val_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    print(
        f"Epoch {epoch + 1}/{epochs}, Training Loss: {running_loss / len(train_loader):.4f}"
    )

    # Evaluate on the test set
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    test_accuracies.append(accuracy)
    val_losses.append(val_running_loss / len(test_loader))
    print(
        f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_running_loss / len(test_loader):.4f}"
    )
    print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}")

# Step 4: Model Evaluation
print("Evaluating the model...")
y_true, y_pred = [], []
summary(model)
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[str(i) for i in range(10)]
)
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(18, 5))

# Training Loss
print("train losses: ", train_losses)
plt.subplot(1, 3, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Validation Loss
print("validation losses: ", val_losses)
plt.subplot(1, 3, 2)
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.legend()

# Test Accuracy
plt.subplot(1, 3, 3)
print("test acc: ", test_accuracies)
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
