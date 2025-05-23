import cv2
import numpy as np
import matplotlib.pyplot as plt


def watershed_segmentation(image_path):
    # 1. Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None, None

    # Преобразование в градации серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Бинаризация и удаление шума
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Фон и объекты
    sure_bg = cv2.dilate(opening, kernel, iterations=15)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)

    # 4. Метки
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 5. Водораздел
    img_ws = img.copy()
    cv2.watershed(img_ws, markers)
    img_ws[markers == -1] = [0, 0, 255]

    # 6. Подсчёт размеров сегментов
    unique_labels = np.unique(markers)
    segment_sizes = []
    for label in unique_labels:
        if label <= 0:  # Пропускаем фон и границы
            continue
        size = np.sum(markers == label)  # Количество пикселей для текущего сегмента
        segment_sizes.append(size)

    num_segments = len(segment_sizes)  # Количество сегментов

    return img_ws, markers, num_segments, segment_sizes


# Список изображений для обработки
path = "../3/cat.jpg"
img_ws, markers, num_segments, segment_sizes = watershed_segmentation(path)

# Визуализация результатов
if img_ws is not None:
    img = cv2.imread(path)  # Загружаем оригинальное изображение для отображения
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Оригинал")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_ws, cv2.COLOR_BGR2RGB))
    plt.title("Watershed Segmentation")
    plt.axis('off')
    plt.tight_layout()
    plt.show()