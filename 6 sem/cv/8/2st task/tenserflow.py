import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка набора данных Sign Language MNIST (предполагается, что CSV-файлы загружены с Kaggle)
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

# Нормализация значений пикселей в диапазон [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Преобразование меток в one-hot формат
y_train = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5)


# Функция для создания и обучения модели
def train_model(hidden_layers, dropout_rate=0.5, epochs=50):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(28, 28)))  # Преобразование изображения 28x28 в вектор
    for size in hidden_layers:
        model.add(Dense(size, activation='relu'))  # Добавление полносвязного слоя с ReLU
        model.add(Dropout(dropout_rate))  # Добавление слоя dropout для регуляризации
    model.add(Dense(5, activation='softmax'))  # Выходной слой для 5 классов

    # Компиляция модели
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Настройка ранней остановки
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Обучение модели
    history = model.fit(X_train.reshape(-1, 28, 28), y_train,
                        epochs=epochs,
                        batch_size=64,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=0)

    # Оценка модели на тестовых данных
    loss, accuracy = model.evaluate(X_test.reshape(-1, 28, 28), y_test, verbose=0)
    return accuracy, history


# Эксперименты с различными архитектурами
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
    accuracy, history = train_model(layers)
    results.append((name, accuracy))

    # Построение графиков точности и потерь
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title(f'Точность (Архитектура: {name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title(f'Потери (Архитектура: {name})')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    #plt.show()

# Вывод результатов
print("Архитектура\tТочность на тесте (%)")
for name, acc in results:
    print(f"{name}\t\t{acc * 100:.2f}")