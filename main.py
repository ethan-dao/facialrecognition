import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),  # Randomly rotate images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pre-trained models
])

# Dataset using image folders in Pytorch
train_dataset = ImageFolder(root = 'fer2013data/train', transform = transform)
test_dataset = ImageFolder(root = 'fer2013data/test', transform = transform)

# Split training data into training and validation (80-20 split)
train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size
train_data, validation_data = random_split(train_dataset, [train_size, validation_size])

# Create data loaders
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
validation_loader = DataLoader(validation_data, batch_size = 64, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

# Choose device
device = (
"cuda"
if torch.cuda.is_available()
else "mps"
if torch.backends.mps.is_available()
else "cpu"
)
print(f"Using {device} device")

# Load pre-trained ResNet model
# model = models.resnet18(pretrained=True)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modify the final fully connected layer to match the number of classes (7 for FER-2013)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # 7 classes for FER-2013

model = model.to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter('runs/FER2013_ResNet')

# Training loop
def train(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            writer.add_scalar('Training Loss', running_loss / (batch + 1), epoch * len(train_loader) + batch)
        
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {running_loss/len(train_loader)}')
        
        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / len(val_loader.dataset)
        print(f'Validation Loss: {val_loss}, Accuracy: {accuracy}%')

        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', accuracy, epoch)

        # Step the learning rate scheduler
        scheduler.step()

# Evaluation loop
def evaluate(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / len(test_loader.dataset)
    print(f'Test Loss: {test_loss/len(test_loader)}, Accuracy: {accuracy}%')

    writer.add_scalar('Test Loss', test_loss / len(test_loader), 0)
    writer.add_scalar('Test Accuracy', accuracy, 0)

# Run training and evaluation
train(model, train_loader, validation_loader, loss_fn, optimizer, scheduler, epochs=5)
evaluate(model, test_loader, loss_fn)

writer.close()
