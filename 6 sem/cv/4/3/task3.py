import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r'D:\Study\uni\university\6 sem\cv\3\cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny with different thresholds
edges1 = cv2.Canny(gray, 100, 200)  # Standard thresholds
edges2 = cv2.Canny(gray, 50, 150)   # Lower thresholds


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Canny (100, 200)')
plt.imshow(edges1, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Canny (50, 150)')
plt.imshow(edges2, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('canny_edges.png')
plt.show()


cv2.imwrite('canny_edges_100_200.png', edges1)
cv2.imwrite('canny_edges_50_150.png', edges2)