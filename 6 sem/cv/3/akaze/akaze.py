import cv2
import matplotlib.pyplot as plt

# Загрузка изображения
img = cv2.imread('../cat.jpg')

# Преобразование в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Создание объекта AKAZE с подобранными параметрами
akaze = cv2.AKAZE_create(threshold=0.003)

# Обнаружение ключевых точек и описание дескрипторов
kp, des = akaze.detectAndCompute(gray, None)

# Визуализация ключевых точек
img_akaze = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Сохранение результата в файл
cv2.imwrite('AKAZE_Keypoints.jpg', img_akaze)

# Отображение результата
plt.figure(figsize=(10, 5))
plt.title("AKAZE Keypoints")
plt.imshow(cv2.cvtColor(img_akaze, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()