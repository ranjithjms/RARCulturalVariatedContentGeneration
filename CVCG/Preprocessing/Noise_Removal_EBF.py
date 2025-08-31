import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Apply bilateral filter
# Parameters: bilateralFilter(source, diameter, sigmaColor, sigmaSpace)
filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


