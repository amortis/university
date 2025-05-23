import cv2
import matplotlib.pyplot as plt

# Загрузка изображения
img = cv2.imread('../cat.jpg')

# Преобразование в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Создание объекта BRISK с подобранными параметрами
brisk = cv2.BRISK_create(thresh=65, octaves=5, patternScale=1.2)

# Обнаружение ключевых точек и описание дескрипторов
kp, des = brisk.detectAndCompute(gray, None)

# Визуализация ключевых точек
img_brisk = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Сохранение результата в файл
cv2.imwrite('BRISK_Keypoints.jpg', img_brisk)

# Отображение результата
plt.figure(figsize=(10, 5))
plt.title("BRISK Keypoints")
plt.imshow(cv2.cvtColor(img_brisk, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()