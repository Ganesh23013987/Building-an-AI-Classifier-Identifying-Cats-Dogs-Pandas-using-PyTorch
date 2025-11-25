# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-using-PyTorch


Building an AI Classifier: Identifying Cats, Dogs & Pandas using PyTorch

This project demonstrates how to build an image classifier that can identify Cats, Dogs, and Pandas using Transfer Learning with ResNet18 and VGG19.

The experiment was implemented in Google Colab and trained on images stored in Google Drive.

## Project Highlights

Uses PyTorch & Torchvision

Uses Transfer Learning (ResNet18 + VGG19)

Training, validation, testing data split

Uses ImageFolder and a Custom Test Dataset

Includes transformations, normalization & augmentation

Tracks training + validation accuracy

Saves best_model_resnet18.pth

Predicts & displays results on random test images

## 📂 Dataset Structure

Make sure your dataset looks like this:

Cat-Dog_Pandas/
│── Train/
│     ├── Cat/
│     ├── Dog/
│     └── Panda/
│
│── Valid/
│     ├── Cat/
│     ├── Dog/
│     └── Panda/
│
└── Test/
      ├── image1.jpg
      ├── image2.jpg
      └── ...

## 🚀 How to Run the Project (Google Colab)
1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

2. Install Libraries
!pip install torch torchvision

3. Check GPU
import torch
print("CUDA available:", torch.cuda.is_available())

## 🧠 Training Using ResNet18

Load ResNet18 pre-trained model

Freeze convolution layers

Replace the fully-connected layer

Train only the classifier

## 📌 Model Training Output

The notebook prints:

Training Loss & Accuracy

Validation Loss & Accuracy

Best Accuracy achieved

Saved model: best_model_resnet18.pth

## 🖼️ Prediction Visualization

Displays random test images with predicted labels.

## 🔥 Bonus: VGG19 Model Included

A second model (VGG19) is prepared for training and comparison.

📄 Output Files

After training, the following file is generated:

best_model_resnet18.pth


Upload this file to GitHub under a folder called models.

## 📘 Tech Stack

Python

PyTorch

Torchvision

Google Colab

Matplotlib

Sklearn

## 🙌 Author

GANESH D

AI & Deep Learning Learner

## requirements.txt 
torch
torchvision
numpy
matplotlib
pandas
seaborn
scikit-learn
Pillow
tqdm

## main.ipynb

```
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
from google.colab import drive
drive.mount('/content/drive')

!lr content/drive/
import torch, torch.nn as nn, torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt, numpy as np, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os, time, copy, pandas as pd
from glob import glob
data_dir = "/content/drive/MyDrive/Cat-Dog_Pandas"  # adjust if needed
train_dir = os.path.join(data_dir, "Train")
test_dir  = os.path.join(data_dir, "Test")
val_dir = os.path.join(data_dir, "Valid")
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds = datasets.ImageFolder(val_dir, transform=test_tfms)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

class_names = train_ds.classes
print("Classes:", class_names)
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
import torch
import torch.nn as nn
from torchvision import models

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze existing layers
for param in model.parameters():
    param.requires_grad = False

# Replace final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(class_names))  # your number of classes
)

# Move to device
model = model.to(device)

print(model)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import os

# Paths
train_dir = "/content/drive/MyDrive/Cat-Dog_Pandas/Train"
val_dir = "/content/drive/MyDrive/Cat-Dog_Pandas/Valid"
test_dir = "/content/drive/MyDrive/Cat-Dog_Pandas/Test"

# Transforms
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Normal datasets
train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds = datasets.ImageFolder(val_dir, transform=test_tfms)
class_names = train_ds.classes
print("Classes:", class_names)

# Custom dataset for test (no subfolders)
class CustomTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.files.extend(glob.glob(os.path.join(test_dir, ext)))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_path  # return path to identify later

test_ds = CustomTestDataset(test_dir, transform=test_tfms)

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

print(f"Loaded {len(train_ds)} train, {len(val_ds)} valid, {len(test_ds)} test images.")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Load pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze conv layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, len(class_names))
)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss, running_corrects = 0.0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects.double() / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] - "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_resnet18.pth")

print("Training complete. Best val acc:", best_val_acc)

import matplotlib.pyplot as plt
import random

# Load best model
model.load_state_dict(torch.load("best_model_resnet18.pth"))
model.eval()

def imshow(img, title=None):
    img = img.cpu().numpy().transpose((1, 2, 0))
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')

# Show random test predictions
fig = plt.figure(figsize=(12, 8))
rows, cols = 3, 3
for i in range(9):
    idx = random.randint(0, len(test_ds)-1)
    img, path = test_ds[idx]
    inputs = img.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
    label = class_names[preds]
    ax = plt.subplot(rows, cols, i+1)
    imshow(img, title=f"Pred: {label}")
plt.tight_layout()
plt.show()

# Define class_names before using it
# Example: if your dataset has 'cat' and 'dog'
class_names = ['cat', 'dog']

from torchvision import models
import torch.nn as nn

# Define VGG19 model and modify classifier
VGG19model = models.vgg19(pretrained=True)
num_ftrs = VGG19model.classifier[6].in_features
VGG19model.classifier[6] = nn.Linear(num_ftrs, len(class_names))

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VGG19model = VGG19model.to(device)

# Load trained weights if available
# VGG19model.load_state_dict(torch.load('best_vgg19_model.pth'))
```

## OUTPUT:



## RESULT:
Thus, the Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-using-PyTorch is implemented and successfully executed.
