import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

# Apply Gaussian filter
# Parameters: GaussianBlur(source, kernel size, standard deviation)
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
