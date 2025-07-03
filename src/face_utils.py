import cv2
import numpy as np

# Load the OpenCV pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face_opencv(image_path, output_size=(224, 224)):
    """
    Detects the largest face in an image and returns a cropped and resized version.

    Args:
        image_path (str): Path to the input image.
        output_size (tuple): Desired output size (width, height).

    Returns:
        np.ndarray: Cropped and resized face image in RGB format, or None if no face is found.
    """
    # Load the image in grayscale and color
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("❌ No face detected.")
        return None

    # Pick the largest face (best guess for main face)
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face = img_color[y:y+h, x:x+w]

    # Resize and convert to RGB
    face = cv2.resize(face, output_size)
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    return face_rgb
