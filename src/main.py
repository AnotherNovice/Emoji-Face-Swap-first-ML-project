
from deepface import DeepFace
import cv2
import numpy as np
import streamlit as st
import sys


@st.cache_resource
def load_models():
    """Pre-load DeepFace models so they're ready when needed!"""
    # This forces the model to download and initialize
    try:
        # A dummy prediction that triggers model loading
        DeepFace.verify(
            img1_path="placeholder",
            img2_path="placeholder",
            model_name="VGG-Face",
            enforce_detection=False
        )
    except:
        pass  # We expect this to fail—we just want the download!

    return True

# Call it immediately
load_models()

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
    max_width = 800
    scale = max_width / img.shape[1]
    display_img = cv2.resize(img, None, fx=scale, fy=scale)
    cv2.imshow("Emoji Face Swap", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #run with arg of path to picture
    image = None

    try:
        image = sys.argv[1]
    except IndexError:
        print("Usage: python3 main.py image_path")
    except:
        print("no bueno")

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
        "happy": "Data/emojis/happy.png",
        "sad": "Data/emojis/sad.png",
        "angry": "Data/emojis/angry.png",
        "surprise": "Data/emojis/surprise.png",
        "fear": "Data/emojis/fear.png",
        "disgust": "Data/emojis/disgust.png",
        "neutral": "Data/emojis/neutral.png"
    }

    emoji_path = emotion_to_path[emotion]

    overlay_emoji_on_face(image, emoji_path, face_box)

    emoji = emotion_to_emoji.get(emotion, "❓")
    print(emoji)
