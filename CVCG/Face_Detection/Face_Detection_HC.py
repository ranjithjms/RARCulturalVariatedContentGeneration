import cv2

class Facial_Object_Detection_HC:
    def detect_facial_objects(self, spath, dpath):
        # Load the pre-trained Haar cascades for face, eyes, nose, and mouth detection
        face_cascade = cv2.CascadeClassifier('..\\Files\\haarcascade_frontalface_default.xml')
        # eye_cascade = cv2.CascadeClassifier('..\\Files\\haarcascade_eye.xml')
        # nose_cascade = cv2.CascadeClassifier('..\\Files\\haarcascade_mcs_nose.xml')  # You may need to download this
        # mouth_cascade = cv2.CascadeClassifier('..\\Files\\haarcascade_mcs_mouth.xml')  # You may need to download this

        # Read the image
        image = cv2.imread(spath)

        # Convert the image to grayscale for the classifier
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces)>0:
            # Loop over each detected face and then detect eyes, nose, and mouth within the face region
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Get the region of interest (ROI) for the face
                face_roi_gray = gray_image[y:y + h, x:x + w]
                face_roi_color = image[y:y + h, x:x + w]

                # Detect eyes within the face ROI
                # eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
                # for (ex, ey, ew, eh) in eyes:
                #     cv2.rectangle(face_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                # Detect nose within the face ROI
                # noses = nose_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                # for (nx, ny, nw, nh) in noses:
                #     cv2.rectangle(face_roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)

                # Detect mouth within the face ROI (usually below the nose, so y+h is adjusted)
                # mouths = mouth_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))
                # for (mx, my, mw, mh) in mouths:
                    # Ensure mouth is below the nose (optional filtering based on position)
                    # if my > h // 2:  # Mouth is typically in the lower half of the face
                    #     cv2.rectangle(face_roi_color, (mx, my), (mx + mw, my + mh), (255, 0, 255), 2)

            cv2.imwrite(dpath, image)