import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import threading

class FacialApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial")
        self.root.geometry("600x400")
        
        # Variables de control
        self.capturing = False
        self.recognizing = False
        
        self.load_images()
        self.create_widgets()
        self.check_folders()
        
    def load_images(self):
        try:
            self.capture_img = ImageTk.PhotoImage(Image.open("icons/capture_icon.png").resize((64,64)))
            self.train_img = ImageTk.PhotoImage(Image.open("icons/train_icon.png").resize((64,64)))
            self.recognize_img = ImageTk.PhotoImage(Image.open("icons/recognize_icon.png").resize((64,64)))
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando imágenes: {str(e)}")
            self.root.destroy()

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TButton", font=('Arial', 10), padding=10)
        
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=40)
        
        # Botones con iconos
        self.btn_capturar = ttk.Button(
            button_frame,
            image=self.capture_img,
            text="Capturar Rostros",
            compound=tk.TOP,
            command=self.start_capture
        )
        self.btn_capturar.grid(row=0, column=0, padx=20)
        
        self.btn_entrenar = ttk.Button(
            button_frame,
            image=self.train_img,
            text="Entrenar Modelo",
            compound=tk.TOP,
            command=self.start_training
        )
        self.btn_entrenar.grid(row=0, column=1, padx=20)
        
        self.btn_reconocer = ttk.Button(
            button_frame,
            image=self.recognize_img,
            text="Reconocimiento Facial",
            compound=tk.TOP,
            command=self.start_recognition
        )
        self.btn_reconocer.grid(row=0, column=2, padx=20)
        
        # Barra de estado
        self.status_bar = ttk.Label(
            self.root, 
            text="Estado: Listo", 
            relief=tk.SUNKEN,
            padding=5
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def check_folders(self):
        if not os.path.exists("rostros"):
            os.makedirs("rostros")
            
    def update_status(self, message):
        self.status_bar.config(text=f"Estado: {message}")
        self.root.update()

    # ------------------- Funcionalidad integrada -------------------
    def start_capture(self):
        if self.capturing:
            return
            
        nombre = simpledialog.askstring("Input", "Ingrese el nombre del nuevo usuario:")
        if nombre:
            self.capturing = True
            threading.Thread(target=self.capture_faces, args=(nombre,), daemon=True).start()

    def capture_faces(self, nombre):
        self.update_status("Capturando rostros...")
        ruta_guardado = f"rostros/{nombre}"
        os.makedirs(ruta_guardado, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 0
        
        while count < 100 and self.capturing:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
                rostro = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                cv2.imwrite(f"{ruta_guardado}/{count}.jpg", rostro)
                count += 1
            
            cv2.imshow("Capturando Rostros", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.capturing = False
        self.update_status("Captura completada")

    def start_training(self):
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        self.update_status("Entrenando modelo...")
        try:
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
            face_recognizer = cv2.face.LBPHFaceRecognizer.create()  # Línea modificada
            face_recognizer.train(faces_data, labels)
            face_recognizer.write("modelo_rostros.xml")
        
            messagebox.showinfo("Éxito", "Modelo entrenado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}\nRevisa la instalación de OpenCV")
        self.update_status("Listo")

    def start_recognition(self):
        if self.recognizing:
            return
            
        self.recognizing = True
        threading.Thread(target=self.face_recognition, daemon=True).start()

    def face_recognition(self):
        self.update_status("Reconociendo...")
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read("modelo_rostros.xml")
    
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        cap = cv2.VideoCapture(0)
        people = sorted(os.listdir("rostros"))  # Ordenar alfabéticamente
    
        while self.recognizing:
            ret, frame = cap.read()
            if not ret:
                break
        
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
            for (x, y, w, h) in faces:
                rostro = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label, confidence = face_recognizer.predict(rostro)
            
                nombre = "Desconocido"
                color = (0, 0, 255)
            
                if confidence < 120:
                    # Verificar que la etiqueta esté dentro del rango
                    if label < len(people):
                        nombre = people[label]
                        color = (0, 255, 0)
            
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
            cv2.imshow("Reconocimiento Facial", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()
        self.recognizing = False
        self.update_status("Listo")

    def on_close(self):
        self.capturing = False
        self.recognizing = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Ahora sí existe el método
    root.mainloop()