## where the magic happens
from src.face_utils import detect_and_crop_face_opencv
import matplotlib.pyplot as plt

face = detect_and_crop_face_opencv(r"C:\Users\ethan\PycharmProjects\PythonProject\Data\faces\SmileWoman.jpg")

if face is not None:
    plt.imshow(face)
    plt.axis('off')
    plt.show()
