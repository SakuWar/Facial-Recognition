import cv2
import os

# Nombre del usuario
usuario = input("Ingrese el nombre del nuevo usuario: ")
ruta_guardado = f"rostros/{usuario}"

# Crear la carpeta si no existe
if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)

# Iniciar la cámara
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

contador = 0

while contador < 100:  # Capturar 100 imágenes
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = frame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (200, 200))
        cv2.imwrite(f"{ruta_guardado}/{contador}.jpg", rostro)
        contador += 1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    cv2.imshow("Capturando Rostros", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
