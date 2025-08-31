import cv2
#Viola Jones
class Existing_VJ:
    def object_segmentation(self, spath, dpath):
        # Load the pre-trained Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '..\\Files\\haarcascade_frontalface_default.xml')

        # Function to detect faces in an image
        def detect_faces(image):
            # Convert the image to grayscale (Haar Cascade works better on gray images)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            return image

        # Load an image
        image_path = spath  # Replace with your image path
        image = cv2.imread(image_path)

        # Detect faces in the image
        output_image = detect_faces(image)

        # Display the output image
        cv2.imshow('Detected Faces', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optional: Save the output image
        cv2.imwrite(dpath, output_image)
