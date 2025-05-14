import cv2
import matplotlib.pyplot as plt

# Загрузка изображения
img = cv2.imread('../cat.jpg')

# Преобразование в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Создание объекта ORB с новыми параметрами
# nfeatures=5 количество ключевых точек
# edgeThreshold=31 порог для отсечения угловых точек
orb = cv2.ORB_create(nfeatures=5, edgeThreshold=31, scoreType=cv2.ORB_HARRIS_SCORE)

# Обнаружение ключевых точек и описание дескрипторов
kp, des = orb.detectAndCompute(gray, None)

# Визуализация ключевых точек
img_orb = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Сохранение результата в файл
cv2.imwrite('ORB_Keypoints.jpg', img_orb)

# Отображение результата
plt.figure(figsize=(10, 5))
plt.title("ORB Keypoints")
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()