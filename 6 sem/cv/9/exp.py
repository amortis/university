
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from uuid import uuid4
import pandas as pd
from torch.optim.lr_scheduler import StepLR

# Устанавливаем фиксированное начальное значение для воспроизводимости результатов
torch.manual_seed(42)
# Определяем устройство (GPU, если доступно, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")














# Определяем гибкий класс сверточной нейронной сети (CNN) для экспериментов с архитектурой
class FlexibleCNN(nn.Module):
    def __init__(self, num_filters, kernel_size, num_conv_layers, dense_units, num_dense_layers, activation,
                 dropout_rate=0.0):
        super(FlexibleCNN, self).__init__()
        self.layers = nn.ModuleList()

        # Входные данные: изображения 28x28x1 (MNIST, оттенки серого)
        in_channels = 1

        # Добавляем сверточные слои
        for i in range(num_conv_layers):
            filters = num_filters[i] if i < len(num_filters) else num_filters[-1]
            self.layers.append(nn.Conv2d(in_channels, filters, kernel_size, padding="same"))
            self.layers.append(self.get_activation(activation))
            self.layers.append(nn.MaxPool2d(2, stride=2))
            if dropout_rate > 0:
                self.layers.append(nn.Dropout2d(dropout_rate))
            in_channels = filters

        # Слой выравнивания (преобразование тензора в вектор)
        self.flatten = nn.Flatten()

        # Вычисляем размер выхода после сверточных слоев (28x28 уменьшается в 2 раза за каждый пуллинг)
        conv_output_size = 28 // (2 ** num_conv_layers)
        dense_input_size = in_channels * conv_output_size * conv_output_size

        # Добавляем полносвязные слои
        self.dense_layers = nn.ModuleList()
        current_units = dense_input_size
        for i in range(num_dense_layers):
            next_units = dense_units[i] if i < len(dense_units) else dense_units[-1]
            self.dense_layers.append(nn.Linear(current_units, next_units))
            self.dense_layers.append(self.get_activation(activation))
            if dropout_rate > 0 and i < num_dense_layers - 1:  # Без dropout на последнем слое
                self.dense_layers.append(nn.Dropout(dropout_rate))
            current_units = next_units

        # Выходной слой (10 классов для MNIST)
        self.dense_layers.append(nn.Linear(current_units, 10))
        self.dense_layers.append(nn.Softmax(dim=1))

    def get_activation(self, activation_name):
        """Возвращает указанную функцию активации"""
        if activation_name.lower() == "relu":
            return nn.ReLU()
        elif activation_name.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            return nn.Tanh()
        elif activation_name.lower() == "elu":
            return nn.ELU()
        elif activation_name.lower() == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Неподдерживаемая функция активации: {activation_name}")

    def forward(self, x):
        # Проход через сверточные слои
        for layer in self.layers:
            x = layer(x)
        # Выравнивание
        x = self.flatten(x)
        # Проход через полносвязные слои
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def count_parameters(self):
        """Подсчитывает общее количество обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Функция для аугментации данных (Задание 4)
def get_transforms(augmentation=True):
    if augmentation:
        return transforms.Compose([
            transforms.RandomRotation(10),  # Случайный поворот на ±10 градусов
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Случайный сдвиг
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Нормализация для MNIST
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


# Загрузка датасета MNIST
def load_mnist_data(batch_size, augmentation=False):
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=get_transforms(augmentation))
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                              transform=get_transforms(False))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Класс для ранней остановки обучения
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
        self.patience = patience  # Количество эпох без улучшения
        self.delta = delta  # Минимальное улучшение для сброса счетчика
        self.best_loss = float('inf')  # Лучшая потеря
        self.counter = 0  # Счетчик эпох без улучшения
        self.best_model_state = None  # Лучшее состояние модели

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
        return self.counter >= self.patience


# Функция обучения модели
def train_model(model, train_loader, val_loader, optimizer_name, learning_rate, l2_lambda, epochs, early_stopping):
    criterion = nn.CrossEntropyLoss()  # Функция потерь
    # Выбор оптимизатора
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_lambda)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    elif optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    else:
        raise ValueError(f"Неподдерживаемый оптимизатор: {optimizer_name}")

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # Уменьшение скорости обучения
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}  # История метрик

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # Обнуляем градиенты
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # Обратное распространение
            optimizer.step()  # Обновление весов

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Валидация
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Эпоха {epoch + 1}/{epochs}: Потери на обучении: {train_loss:.4f}, Точность на обучении: {train_acc:.4f}, Потери на валидации: {val_loss:.4f}, Точность на валидации: {val_acc:.4f}")

        # Проверка ранней остановки
        if early_stopping(val_loss, model):
            print("Ранняя остановка сработала")
            model.load_state_dict(early_stopping.best_model_state)
            break

        scheduler.step()

    return history


# Функция оценки модели
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total


# Функция для построения графиков
def plot_history(history, experiment_name):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Потери на обучении')
    plt.plot(history['val_loss'], label='Потери на валидации')
    plt.title(f'Потери по эпохам ({experiment_name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Точность на обучении')
    plt.plot(history['val_acc'], label='Точность на валидации')
    plt.title(f'Точность по эпохам ({experiment_name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{experiment_name}_plot.png')
    plt.close()


# Конфигурации экспериментов
experiments = [
    # Задание 1: Изменение архитектуры
    {"name": "baseline", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "more_filters", "num_filters": [64, 128], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "larger_kernel", "num_filters": [32, 64], "kernel_size": 5, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "more_conv_layers", "num_filters": [32, 64, 128], "kernel_size": 3, "num_conv_layers": 3,
     "dense_units": [128], "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32,
     "optimizer": "adam", "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "more_dense_units", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [256],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "more_dense_layers", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2,
     "dense_units": [128, 64], "num_dense_layers": 2, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32,
     "optimizer": "adam", "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},

    # Задание 2: Эксперименты с функциями активации
    {"name": "sigmoid", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "sigmoid", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "tanh", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "tanh", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "elu", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "elu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "leakyrelu", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "leakyrelu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},

    # Задание 3: Эксперименты с оптимизаторами
    {"name": "sgd", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "sgd", "lr": 0.01,
     "l2_lambda": 0.0, "augmentation": False},
    {"name": "rmsprop", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "rmsprop",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "adagrad", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adagrad",
     "lr": 0.01, "l2_lambda": 0.0, "augmentation": False},
    {"name": "adamw", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adamw",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},

    # Задание 4: Dropout, размер батча, аугментация, L2-регуляризация
    {"name": "dropout_0.3", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.3, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "batch_size_16", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 16, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "batch_size_128", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 128, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": False},
    {"name": "augmentation", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.0, "augmentation": True},
    {"name": "l2_reg_0.01", "num_filters": [32, 64], "kernel_size": 3, "num_conv_layers": 2, "dense_units": [128],
     "num_dense_layers": 1, "activation": "relu", "dropout_rate": 0.0, "batch_size": 32, "optimizer": "adam",
     "lr": 0.001, "l2_lambda": 0.01, "augmentation": False},
]

# Выполнение экспериментов и сбор результатов
results = []
for config in experiments:
    print(f"\nЗапуск эксперимента: {config['name']}")

    # Загрузка данных
    train_loader, test_loader = load_mnist_data(config['batch_size'], config['augmentation'])

    # Разделение обучающего набора на обучение и валидацию
    train_size = int(0.8 * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Инициализация модели
    model = FlexibleCNN(
        num_filters=config['num_filters'],
        kernel_size=config['kernel_size'],
        num_conv_layers=config['num_conv_layers'],
        dense_units=config['dense_units'],
        num_dense_layers=config['num_dense_layers'],
        activation=config['activation'],
        dropout_rate=config['dropout_rate']
    ).to(device)

    # Подсчет параметров
    num_params = model.count_parameters()

    # Инициализация ранней остановки
    early_stopping = EarlyStopping(patience=3)

    # Обучение модели
    history = train_model(
        model, train_loader, val_loader, config['optimizer'], config['lr'],
        config['l2_lambda'], epochs=5, early_stopping=early_stopping
    )

    # Оценка модели
    test_acc = evaluate_model(model, test_loader)

    # Сохранение результатов
    results.append({
        'Эксперимент': config['name'],
        'Число фильтров': config['num_filters'],
        'Размер ядра': config['kernel_size'],
        'Число сверточных слоев': config['num_conv_layers'],
        'Полносвязные нейроны': config['dense_units'],
        'Число полносвязных слоев': config['num_dense_layers'],
        'Функция активации': config['activation'],
        'Dropout': config['dropout_rate'],
        'Размер батча': config['batch_size'],
        'Оптимизатор': config['optimizer'],
        'Скорость обучения': config['lr'],
        'L2-регуляризация': config['l2_lambda'],
        'Аугментация': config['augmentation'],
        'Параметры': num_params,
        'Точность на тесте': test_acc,
        'Финальная потеря на валидации': history['val_loss'][-1],
        'Финальная точность на валидации': history['val_acc'][-1]
    })

    # Построение графиков
    plot_history(history, config['name'])

# Сохранение результатов в таблицу
results_df = pd.DataFrame(results)
results_df.to_csv('experiment_results.csv', index=False)
print("\nРезультаты сохранены в 'experiment_results.csv'")

# Пример предсказания на тестовом изображении
model.eval()
data_iter = iter(test_loader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    output = model(images[0:1])
    predicted = torch.argmax(output, dim=1).item()
    plt.imshow(images[0].cpu().squeeze(), cmap='gray')
    plt.title(f'Предсказано: {predicted}, Фактически: {labels[0].item()}')
    plt.savefig('sample_prediction.png')
    plt.close()

# Примечание для самостоятельной работы (датасет жестов):
# 1. Замените MNIST на датасет жестов, используя torchvision.datasets.ImageFolder или кастомный датасет.
# 2. Измените in_channels=3 (для RGB) в FlexibleCNN и число выходных классов (5 для 5 жестов).
# 3. Настройте get_transforms для размера изображений датасета (например, добавьте transforms.Resize).
# 4. Повторите эксперименты для подбора оптимальной архитектуры.
# Источники датасетов: https://www.kaggle.com/datasets, https://public.roboflow.com
