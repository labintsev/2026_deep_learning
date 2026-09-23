"""
Обучение сверточного классификатора на наборе данных CIFAR-10
"""
import sys
import os

# This tells Python to also look in the folder one level up
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
# create pytorch datasets
trainset = torch.utils.data.TensorDataset(
    torch.tensor(X_train).permute(0, 3, 1, 2).float() / 255.0, torch.tensor(y_train)
    )
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

FEATURE_DIM = 16  # Размерность вектора признаков на предпоследнем слое

# Simple CNN model
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),      
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 8 * 8, FEATURE_DIM),
    nn.ReLU(),
    nn.Linear(FEATURE_DIM, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(5):  # Обучаем модель в течение 5 эпох
    print(f'Epoch {epoch + 1}/5')
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    loss_value = loss.item()
    print(f'Loss: {loss_value:.4f}')

# save the model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/cifar10_cnn.pth')
