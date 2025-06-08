import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Загрузка и предобработка данных
def load_sign_mnist_data():
    train_data = pd.read_csv('./data/sign/sign_mnist_train/sign_mnist_train.csv')
    test_data = pd.read_csv('./data/sign/sign_mnist_test/sign_mnist_test.csv')

    # Оставим только метки 0–4 (A–E)
    train_data = train_data[train_data['label'].isin([0, 1, 2, 3, 4])]
    test_data = test_data[test_data['label'].isin([0, 1, 2, 3, 4])]

    # Уменьшаем тренировочный набор для большей сложности
    train_data = train_data.sample(frac=0.3, random_state=42)

    train_labels = train_data['label'].values
    train_images = train_data.drop('label', axis=1).values
    test_labels = test_data['label'].values
    test_images = test_data.drop('label', axis=1).values

    # Нормализация
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    # One-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=5)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=5)

    # Разделение на тренировочную и валидационную выборки
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

# Упрощённая модель CNN с сильной регуляризацией
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    return model

# Графики обучения
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('Потери по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title('Точность по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('sign_mnist_training_plot_tf.png')
    plt.close()

# Основной код
(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_sign_mnist_data()

# Создание и компиляция модели
model = build_cnn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение
history = model.fit(
    train_images, train_labels,
    epochs=20,
    validation_data=(val_images, val_labels),
    batch_size=32
)

# Оценка
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Тестовые потери: {test_loss:.4f}, Тестовая точность: {test_acc*100:.2f}%')

# Сохранение модели
model.save('sign_mnist_cnn.keras')

# Пример предсказания
first_image = test_images[1]
plt.imshow(first_image.squeeze(), cmap='gray')
pred = model.predict(first_image[np.newaxis, ...])
predicted_class = np.argmax(pred, axis=1)[0]
actual_class = np.argmax(test_labels[1])
plt.title(f'Предсказано: {chr(predicted_class + 65)}, Фактически: {chr(actual_class + 65)}')
plt.savefig('sign_mnist_prediction_tf.png')
plt.close()
