import cv2
import matplotlib.pyplot as plt

def binarize_threshold_rgb(image_path, threshold_value):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def binarize_adaptive_threshold(image_path, blockSize, C):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
    return thresh

# Load image
image_path = r'D:\Study\uni\university\6 sem\cv\3\cat.jpg'

# Apply global thresholding
global_thresh = binarize_threshold_rgb(image_path, 200)

# Apply adaptive thresholding
adaptive_thresh = binarize_adaptive_threshold(image_path, 11, 2)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Оригинал')
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('cv2.threshold')
plt.imshow(global_thresh, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('cv2.adaptiveThreshold')
plt.imshow(adaptive_thresh, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('binarization_comparison.png')
plt.show()

# Save results
cv2.imwrite('global_threshold.png', global_thresh)
cv2.imwrite('adaptive_threshold.png', adaptive_thresh)