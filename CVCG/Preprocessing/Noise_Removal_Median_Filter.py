import cv2

class Noise_Removal_MF:
    def noise_removal(self, spath, dpath):
        # Read the image
        image = cv2.imread(spath, cv2.IMREAD_COLOR)

        # Apply Median Blur (Median Filter)
        # The second parameter (ksize) must be an odd number (e.g., 3, 5, 7)
        median_filtered = cv2.medianBlur(image, ksize=5)

        # Save the noise removed image
        cv2.imwrite(dpath, cv2.cvtColor(median_filtered, cv2.COLOR_BGR2RGB))