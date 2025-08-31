import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Create a 3x3 kernel for a simple low-pass filter (averaging)
kernel = np.ones((3, 3), np.float32) / 9

# Apply the low-pass filter
filtered_image = cv2.filter2D(image, -1, kernel)
