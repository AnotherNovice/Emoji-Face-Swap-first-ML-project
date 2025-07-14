
from deepface import DeepFace
import cv2
import numpy as np


def analyze_emotion(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    # Run emotion detection
    result = DeepFace.analyze(img_path=image_path, actions=["emotion"])
    dominant_emotion = result[0]["dominant_emotion"]
    face_box = result[0]["region"]  # Contains top, left, width, height
    print(f"Detected Emotion: {dominant_emotion}")
    return dominant_emotion, face_box

    # Annotate and display

    #font = cv2.FONT_HERSHEY_SIMPLEX
   # cv2.putText(image, f'Emotion: {dominant_emotion}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
  #  cv2.imshow("Emotion Detected", image)
 #   cv2.waitKey(0)
#    cv2.destroyAllWindows()



def overlay_emoji_on_face(image_path, emoji_path, face_box):
    # Load the input image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Could not load input image")

        # Load emoji image (with alpha channel for transparency)
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        raise FileNotFoundError("Could not load emoji image")

# Resize emoji to match face size
    face_width, face_height = face_box["w"], face_box["h"]
    resized_emoji = cv2.resize(emoji, (face_width, face_height), interpolation=cv2.INTER_AREA)

    # Extract alpha channel
    if resized_emoji.shape[2] == 4:
        alpha_emoji = resized_emoji[:, :, 3] / 255.0
        rgb_emoji = resized_emoji[:, :, :3]
    else:
        alpha_emoji = np.ones((face_height, face_width))
        rgb_emoji = resized_emoji

    # Get face location
    top, left = face_box["y"], face_box["x"]


   # Overlay the emoji on the original image
    for c in range(3):  # For R, G, B
        img[top:top+face_height, left:left+face_width, c] = (
            alpha_emoji * rgb_emoji[:, :, c] +
            (1 - alpha_emoji) * img[top:top+face_height, left:left+face_width, c]
        )

    # Show the result
    cv2.imshow("Emoji Face Swap", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = r"C:\Users\ethan\PycharmProjects\PythonProject\Data\faces\Sad2.jpg"
    emotion, face_box = analyze_emotion(image)
    emotion_to_emoji = {
        "happy": "😄",
        "sad": "😢",
        "angry": "😠",
        "surprise": "😲",
        "fear": "😨",
        "disgust": "🤢",
        "neutral": "😐"
    }

    emotion_to_path = {
        "happy": "data/emojis/happy.png",
        "sad": r"C:\Users\ethan\PycharmProjects\PythonProject\Data\emojis\sad.png",
        "angry": "data/emojis/angry.png",
        "surprise": "data/emojis/surprise.png",
        "fear": "data/emojis/fear.png",
        "disgust": "data/emojis/disgust.png",
        "neutral": "data/emojis/neutral.png"
    }

    emoji_path = emotion_to_path[emotion]

    overlay_emoji_on_face(image, emoji_path, face_box)

    emoji = emotion_to_emoji.get(emotion, "❓")
    print(emoji)
