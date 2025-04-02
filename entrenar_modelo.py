import cv2
import os
import numpy as np

data_path = "rostros"
people = os.listdir(data_path)
labels = []
faces_data = []
label = 0

for person in people:
    person_path = os.path.join(data_path, person)
    for image in os.listdir(person_path):
        img_path = os.path.join(person_path, image)
        img = cv2.imread(img_path, 0)
        faces_data.append(img)
        labels.append(label)
    label += 1

labels = np.array(labels)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces_data, labels)
face_recognizer.write("modelo_rostros.xml")
print("Modelo entrenado y guardado correctamente.")
