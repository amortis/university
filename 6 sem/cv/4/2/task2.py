import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\Study\uni\university\6 sem\cv\3\cat.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel operator
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges


sobel = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2.0), 1.0, cv2.pow(sobely, 2.0), 1.0, 0.0))


_, sobel_thresh = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Оригинал')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Sobel X')
plt.imshow(sobelx, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Sobel Combined')
plt.imshow(sobel_thresh, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('sobel_edges.png')
plt.show()


cv2.imwrite('sobel_edges.png', sobel_thresh)