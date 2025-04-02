# 👤 Facial-Recognition

Un sistema de reconocimiento facial en tiempo real utilizando OpenCV y Haar Cascades.

<p align="center">
  <img src="https://raw.githubusercontent.com/SakuWar/Facial-Recognition/main/Screenshot%202025-04-02%20170628.png" alt="Demo">
</p>

## Características ✨

- Detección facial en tiempo real usando webcam
- Utiliza Haar Cascade para detección de rostros
- Fácil de implementar y modificar
- Visualización con rectángulos alrededor de los rostros detectados

## Prerrequisitos 📋

- Python 3.6 o superior
- OpenCV (`opencv-python`)
- Numpy
- Webcam funcional

## Instalación 🔧

Clona el repositorio:
```bash
git clone https://github.com/SakuWar/Facial-Recognition.git
cd Facial-Recognition
```

## Uso 🚀

Ejecuta el script principal:
```bash
python grafica.py
```
- Presiona 'q' para salir de la aplicación
- Asegúrate de tener buena iluminación para mejores resultados
- El archivo haarcascade_frontalface_default.xml debe estar en el mismo directorio

## Personalización ⚙️

- Puedes usar diferentes clasificadores Haar Cascade cambiando el archivo .xml
- Modifica los parámetros de detección en detectMultiScale
- Ajusta el grosor del rectángulo y color en el código
