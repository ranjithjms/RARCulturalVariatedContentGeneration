import cv2
import numpy as np
#Single-Shot Detector
class Existing_SSD:
    def object_segmentation(self, spath, dpath):
        # Load the pre-trained SSD model and the configuration file
        model_path = '..\\Files\\MobileNetSSD_deploy.caffemodel'  # Replace with your model path
        config_path = '..\\Files\\MobileNetSSD_deploy.prototxt'     # Replace with your config path

        # Load the class labels the model was trained on
        class_labels = ["background", "person", "bicycle", "car", "motorbike", "aeroplane", "bus",
                        "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
                        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                        "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
                        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                        "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                        "toothbrush"]

        # Load the model
        net = cv2.dnn.readNetFromCaffe(config_path, model_path)

        # Function to detect objects in an image
        def detect_objects(image):
            # Get the image dimensions
            (h, w) = image.shape[:2]

            # Prepare the image for the model
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)

            # Perform forward pass to get the detections
            detections = net.forward()

            # Process detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections
                if confidence > 0.2:  # Adjust threshold as needed
                    idx = int(detections[0, 0, i, 1])  # Class index
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = f"{class_labels[idx]}: {confidence:.2f}"
                    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(image, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return image

        # Load an image
        image_path = spath  # Replace with your image path
        image = cv2.imread(image_path)

        # Detect objects in the image
        output_image = detect_objects(image)

        # Optional: Save the output image
        cv2.imwrite(dpath, output_image)
