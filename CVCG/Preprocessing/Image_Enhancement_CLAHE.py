import cv2
import numpy as np

# Lorenz Curve Contrast Limited Adaptive Histogram Equalization (LCCLAHE)
class Image_Enhancement_CLAHE:
    def image_enhancement(self, spath, dpath):
        # Read the input color image
        image = cv2.imread(spath)

        # Convert the image from BGR to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into its channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L-channel (lightness)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        l_channel_clahe = clahe.apply(l_channel)

        # Merge the CLAHE enhanced L-channel back with the original A and B channels
        lab_image_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

        # Convert the LAB image back to BGR color space
        bgr_image_clahe = cv2.cvtColor(lab_image_clahe, cv2.COLOR_LAB2BGR)

        # Save the enhanced image
        cv2.imwrite(dpath, bgr_image_clahe)