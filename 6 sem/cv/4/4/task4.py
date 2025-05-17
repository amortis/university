import cv2
import matplotlib.pyplot as plt

def apply_sobel(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2.0), 1.0, cv2.pow(sobely, 2.0), 1.0, 0.0))
    _, sobel_thresh = cv2.threshold(sobel, 100, 255, cv2.THRESH_BINARY)
    return sobel_thresh

def apply_canny(image_path, low, high):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, low, high)
    return edges

# Image paths
landscape_path = 'landscape.jpg'
object_path = 'object.jpg'

# Apply operators
sobel_landscape = apply_sobel(landscape_path)
canny_landscape = apply_canny(landscape_path, 100, 200)
sobel_object = apply_sobel(object_path)
canny_object = apply_canny(object_path, 100, 200)

# Display results
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title('Пейзаж оригинал')
img = cv2.imread(landscape_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Пейзаж  Sobel')
plt.imshow(sobel_landscape, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Пейзаж  Canny')
plt.imshow(canny_landscape, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Oбъект оригинал')
img = cv2.imread(object_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Oбъект Sobel')
plt.imshow(sobel_object, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Oбъект Canny')
plt.imshow(canny_object, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('sobel_canny_comparison.png')
plt.show()


cv2.imwrite('sobel_landscape.png', sobel_landscape)
cv2.imwrite('canny_landscape.png', canny_landscape)
cv2.imwrite('sobel_object.png', sobel_object)
cv2.imwrite('canny_object.png', canny_object)