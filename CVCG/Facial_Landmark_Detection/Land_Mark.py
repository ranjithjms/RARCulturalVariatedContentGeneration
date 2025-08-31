import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

class Land_Mark:
    def detect_landmark(self, spath, dpath):
        mp_face_mesh = mp.solutions.face_mesh

        # Initialize the face mesh model
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

        # Load the image
        image_path = spath
        img = cv2.imread(image_path)

        # Convert the BGR image to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and find face landmarks
        result = face_mesh.process(rgb_img)

        # Draw landmarks on the face
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, _ = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

        cv2.imwrite(dpath, img)