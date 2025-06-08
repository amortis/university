
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Устройство: GPU, если доступно
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Кастомный датасет для Sign Language MNIST
class SignMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Загрузка CSV файла
        data = pd.read_csv(csv_file)
        # Фильтрация: выбираем только первые 5 классов (0-4, A-E)
        data = data[data['label'].isin([0, 1, 2, 3, 4])]

        self.labels = data['label'].values
        # Пиксели (исключаем колонку label)
        self.images = data.drop('label', axis=1).values
        self.images = self.images.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0  # [num_samples, 1, 28, 28]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]  # [1, 28, 28]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32)  # Преобразуем в тензор
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# Загрузка данных
def load_sign_mnist_data(batch_size=32):
    # Трансформации для аугментации (только для обучения)
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Загрузка данных
    train_dataset = SignMNISTDataset(csv_file='./data/sign/sign_mnist_train/sign_mnist_train.csv', transform=train_transform)
    test_dataset = SignMNISTDataset(csv_file='./data/sign/sign_mnist_test/sign_mnist_test.csv', transform=test_transform)

    # Разделение обучающего набора на обучение и валидацию (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# Определение модели CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 канал (оттенки серого)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28x28 -> 7x7 после двух пулингов
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 5)  # 5 классов (A-E)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout3(x)
        x = self.softmax(self.fc2(x))
        return x


# Обучение модели
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Валидация
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Эпоха {epoch + 1}, Потери: {epoch_loss:.4f}, Точность: {epoch_acc:.2f}%, '
              f'Валидационные потери: {val_loss:.4f}, Валидационная точность: {val_acc:.2f}%')

    return history


# Оценка модели
def evaluate_model(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return loss, accuracy


# Построение графиков
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Потери на обучении')
    plt.plot(history['val_loss'], label='Потери на валидации')
    plt.title('Потери по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Точность на обучении')
    plt.plot(history['val_acc'], label='Точность на валидации')
    plt.title('Точность по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('sign_mnist_training_plot.png')
    plt.close()


# Основной код
train_loader, val_loader, test_loader = load_sign_mnist_data()

# Инициализация модели
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Обучение
history = train_model(model, train_loader, val_loader, criterion, optimizer)

# Оценка на тесте
test_loss, test_acc = evaluate_model(model, test_loader, criterion)
print(f'Тестовые потери: {test_loss:.4f}, Тестовая точность: {test_acc:.2f}%')

# Сохранение модели
torch.save(model.state_dict(), 'sign_mnist_cnn.pth')

# Загрузка модели
model.load_state_dict(torch.load('sign_mnist_cnn.pth'))

# Визуализация предсказания
model.eval()
images, labels = next(iter(test_loader))
first_image = images[1].cpu().squeeze().numpy()
first_image = (first_image * 0.5 + 0.5).clip(0, 1)  # Денормализация
plt.imshow(first_image, cmap='gray')
with torch.no_grad():
    pred = model(images[1].unsqueeze(0).to(device))
    predicted_digit = torch.argmax(pred, dim=1).item()
plt.title(f'Предсказано: {chr(predicted_digit + 65)}, Фактически: {chr(labels[1].item() + 65)}')
plt.savefig('sign_mnist_prediction.png')
plt.close()
