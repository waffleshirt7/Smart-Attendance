import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []
ids = []

dataset_path = "dataset"

label = 0
label_dict = {}

for folder in os.listdir(dataset_path):
    label_dict[label] = folder
    folder_path = os.path.join(dataset_path, folder)

    for image in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image)
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(gray)
        ids.append(label)

    label += 1

recognizer.train(faces, np.array(ids))
recognizer.save("trainer/trainer.yml")

print("Model trained successfully.")