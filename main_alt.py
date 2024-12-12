import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 5


# Define CNN Model using modern practices
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Modern CNN design with batch normalization after conv layers for better stability
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization improves convergence
        self.pool1 = nn.MaxPool2d(2, 2)  # Max-pooling reduces spatial dimensions

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Sequential execution with ReLU activations
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Training function
def train_model(model, train_loader, val_loader, epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    optimizer = Adam(model.parameters(), lr=lr)  # Adam optimizer for adaptive learning
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))

        # Validation step
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                val_running_loss += loss.item()
        val_loss.append(val_running_loss / len(val_loader))

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}"
        )
    return train_loss, val_loss


# Evaluate on test data
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[str(i) for i in range(10)]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    # Data transforms with normalization
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    # Load MNIST
    mnist_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Train-validation split (80-20 split)
    train_size = int(0.8 * len(mnist_train))
    val_size = len(mnist_train) - train_size
    train_data, val_data = random_split(mnist_train, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

    # Model initialization
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = CNN().to(device)

    # Training and validation
    train_loss, val_loss = train_model(model, train_loader, val_loader)

    # Plotting loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

    # Model evaluation
    evaluate_model(model, test_loader)
