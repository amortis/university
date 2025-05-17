import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Загрузка набора данных Sign Language MNIST
train_data = pd.read_csv('sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist_test.csv')

# Фильтрация данных для классов A, B, C, D, E (метки 0, 1, 2, 3, 4)
selected_classes = [0, 1, 2, 3, 4]
train_data = train_data[train_data['label'].isin(selected_classes)]
test_data = test_data[test_data['label'].isin(selected_classes)]

# Извлечение признаков и меток
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Нормализация и преобразование в тензоры
X_train = torch.tensor(X_train / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test = torch.tensor(y_test, dtype=torch.long)

# Создание DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Определение модели с настраиваемыми скрытыми слоями
class Net(nn.Module):
    def __init__(self, hidden_sizes, dropout_rate=0.5):
        super(Net, self).__init__()
        layers = []
        input_size = 784
        for size in hidden_sizes:
            layers.append(nn.Linear(input_size, size))  # Полносвязный слой
            layers.append(nn.ReLU())  # Активация ReLU
            layers.append(nn.Dropout(dropout_rate))  # Dropout для регуляризации
            input_size = size
        layers.append(nn.Linear(input_size, 5))  # Выходной слой для 5 классов
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 784)  # Преобразование изображения в вектор
        return self.network(x)


# Функция обучения с ранней остановкой
def train_with_early_stopping(hidden_sizes, train_loader, test_loader, max_epochs=50, patience=3):
    model = Net(hidden_sizes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(max_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()  # Обнуление градиентов
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # Обратное распространение
            optimizer.step()  # Обновление весов

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()

        val_loss /= len(test_loader)
        val_accuracy = 100 * correct_val / total_val

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)

        # Проверка условий ранней остановки
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return history, val_accuracy


# Эксперименты с архитектурами
architectures = [
    ([128], "128"),
    ([256], "256"),
    ([512], "512"),
    ([128, 64], "128-64"),
    ([256, 128], "256-128"),
    ([512, 256, 128], "512-256-128")
]

results = []
for layers, name in architectures:
    history, accuracy = train_with_early_stopping(layers, train_loader, test_loader)
    results.append((name, accuracy))

    # Построение графиков потерь и точности
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Потери на обучении')
    plt.plot(history['val_loss'], label='Потери на валидации')
    plt.title(f'Потери (Архитектура: {name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Точность на обучении')
    plt.plot(history['val_acc'], label='Точность на валидации')
    plt.title(f'Точность (Архитектура: {name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.show()

# Вывод результатов
print("Архитектура\tТочность на тесте (%)")
for name, acc in results:
    print(f"{name}\t\t{acc:.2f}")