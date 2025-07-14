
from deepface import DeepFace
import cv2

def analyze_emotion(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    # Run emotion detection
    result = DeepFace.analyze(img_path=image_path, actions=["emotion"])
    dominant_emotion = result[0]["dominant_emotion"]
    print(f"Detected Emotion: {dominant_emotion}")

    # Annotate and display
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f'Emotion: {dominant_emotion}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Emotion Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_emotion(r"C:\Users\ethan\PycharmProjects\PythonProject\Data\faces\SmileWoman.jpg")
