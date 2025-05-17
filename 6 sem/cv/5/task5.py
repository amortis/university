import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_and_merge(image, min_size, homogeneity_threshold, visualize=False):
    rows, cols = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    visualization = []  # Для хранения промежуточных результатов

    def process_region(x, y, width, height, region_image, level=0):
        nonlocal visualization

        region_mean = np.mean(region_image)
        region_std = np.std(region_image)

        if width <= min_size or height <= min_size or region_std <= homogeneity_threshold:
            segmented_image[y:y+height, x:x+width] = region_mean

            if visualize and level < 3:  # Ограничиваем глубину визуализации
                viz = segmented_image.copy()
                cv2.rectangle(viz, (x,y), (x+width,y+height), (255,255,255), 1)
                visualization.append((viz, f"Level {level}: Merge {width}x{height}"))
        else:
            half_width = width // 2
            half_height = height // 2

            if visualize and level < 3:
                viz = segmented_image.copy()
                cv2.rectangle(viz, (x,y), (x+width,y+height), (150,150,150), 1)
                visualization.append((viz, f"Level {level}: Split {width}x{height}"))

            # Рекурсивная обработка 4 подрегионов
            process_region(x, y, half_width, half_height,
                          region_image[:half_height, :half_width], level+1)
            process_region(x + half_width, y, width - half_width, half_height,
                          region_image[:half_height, half_width:], level+1)
            process_region(x, y + half_height, half_width, height - half_height,
                          region_image[half_height:, :half_width], level+1)
            process_region(x + half_width, y + half_height,
                          width - half_width, height - half_height,
                          region_image[half_height:, half_width:], level+1)

    process_region(0, 0, cols, rows, image)

    if visualize:
        plt.figure(figsize=(15, 8))
        for i, (img, title) in enumerate(visualization[:12]):  # Показываем первые 12 шагов
            plt.subplot(3, 4, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return segmented_image

# Загрузка изображения
image_path = "../4/4/object.jpg"
image_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image_original is None:
    print(f"Error: Could not load image from {image_path}")
    exit()

# Параметры для экспериментов
params = [
    {'min_size': 80, 'threshold': 10},   # Большие регионы, низкая чувствительность
    {'min_size': 40, 'threshold': 20},   # Средние регионы, средняя чувствительность
    {'min_size': 20, 'threshold': 5},    # Маленькие регионы, высокая чувствительность
    {'min_size': 10, 'threshold': 2}     # Очень мелкие регионы
]

# Проведение экспериментов
for i, param in enumerate(params):
    print(f"\nЭксперимент {i+1}: min_size={param['min_size']}, threshold={param['threshold']}")

    segmented = split_and_merge(image_original.copy(),
                              param['min_size'],
                              param['threshold'],
                              visualize=True)

    cv2.imwrite(f'split_merge_exp_{i+1}.jpg', segmented)
    plt.show()