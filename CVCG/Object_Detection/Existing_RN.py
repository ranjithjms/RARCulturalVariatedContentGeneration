import cv2
import matplotlib.pyplot as plt
#Selective Search
class Existing_RN:
    def object_segmentation(self, spath, dpath):
        # Load the image
        image_path = spath  # Replace with your image path
        image = cv2.imread(image_path)
        if image is None:
            print("Image not found.")
            exit()

        # Convert the image to RGB for visualization with Matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize Selective Search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()  # Use Fast mode. Alternatively, switchToSelectiveSearchQuality() for higher accuracy.

        # Perform Selective Search to get region proposals
        rects = ss.process()

        # Draw the top region proposals on the image (limit to 100 for visualization)
        output_image = image_rgb.copy()
        num_show_rects = 100  # Limit the number of displayed proposals

        for i, (x, y, w, h) in enumerate(rects[:num_show_rects]):
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the original and output image with region proposals
        plt.figure(figsize=(10, 10))
        plt.imshow(output_image)
        plt.axis("off")
        plt.savefig(dpath)
