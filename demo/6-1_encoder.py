"""
Сравнение векторов признаков на предпоследнем слое для разных изображений
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

_, _, X_test, y_test = load_CIFAR10(cifar10_dir)

mean_image = np.mean(X_test, axis=0)
X_test -= mean_image

testset = torch.utils.data.TensorDataset(
    torch.tensor(X_test).permute(0, 3, 1, 2).float() / 255.0, torch.tensor(y_test)
    )
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False)

FEATURE_DIM = 16  # Размерность вектора признаков на предпоследнем слое

model = model = nn.Sequential(
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

model.eval()

# Переменная, куда мы сохраним значения предпоследнего слоя
features = {}

# 2. Пишем функцию-хук, которая скопирует данные
def get_features_hook(module, input, output):
    features['layer_output'] = output.detach()

# 3. Находим предпоследний слой и вешаем на него хук
# У ResNet18 предпоследний слой — это avgpool перед финальным fc
model[-3].register_forward_hook(get_features_hook)

# 4. Проверяем работу на случайных данных
dummy_input = torch.randn(1, 3, 32, 32)
output = model(dummy_input)

# В features['layer_output'] теперь лежат нужные вам значения!
print("Форма тензора с предпоследнего слоя:", features['layer_output'].shape)


batch = next(iter(testloader))

# Обрабатываем весь батч одним вызовом модели
images, labels = batch

with torch.no_grad():
    outputs = model(images)
    batch_features = features['layer_output'].detach().cpu().numpy()

batch_size = images.size(0)
images -= images.min()


# Выводим векторы признаков для всех изображений батча
# с точностью до 2 знаков

for i in range(batch_size):
    print(f"Изображение {i + 1}, метка: {labels[i].item()}")

    print("Вектор признаков:", [f"{x:.2f}" for x in batch_features[i]])
    print()

# Визуализируем весь батч
columns = 8
rows = (batch_size + columns - 1) // columns

plt.figure(figsize=(16, 2.5 * rows))

for i in range(batch_size):
    plt.subplot(rows, columns, i + 1)

    image = images[i].permute(1, 2, 0).numpy()
    plt.imshow(np.clip(image, 0, 1))

    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.tight_layout()
plt.show()

