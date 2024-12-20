from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import base64
import io
import os
from PIL import Image

app = Flask(__name__)
CORS(app)

# Cargamos el clasificador Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Filtro de paso de banda para la señal
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Filtro de suavizado gaussiano para reducir el ruido
def apply_smoothing(frame):
    return cv2.GaussianBlur(frame, (5, 5), 0)

# Función para ajustar brillo y contraste de la imagen
def adjust_brightness_contrast(image, brightness=30, contrast=30):
    # Ajuste de brillo y contraste
    image = np.array(image)
    new_image = cv2.convertScaleAbs(image, alpha=contrast/127+1, beta=brightness-contrast)
    return new_image

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Obtenemos la imagen en Base64 desde la solicitud
        data = request.json['image']
        img_bytes = base64.b64decode(data)
        
        # Convertimos la imagen de Base64 a un formato compatible con OpenCV
        image = Image.open(io.BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Mejora de la resolución (redimensionar la imagen para mejorar la calidad)
        frame = cv2.resize(frame, (800, 600))  # Puedes ajustar la resolución según sea necesario

        # Aplicamos el filtro de suavizado
        frame = apply_smoothing(frame)

        # Ajuste de brillo y contraste
        frame = adjust_brightness_contrast(frame, brightness=30, contrast=30)

        # Convertimos la imagen a escala de grises y aplicamos ecualización de histograma
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Mejora el contraste para la detección de rostros

        # Detección de rostros con parámetros ajustados
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,  # Aumenta la precisión (menos falsos positivos)
            minNeighbors=5,   # Ajuste de vecinos mínimos
            minSize=(50, 50), # Tamaño mínimo del rostro
            maxSize=(500, 500) # Tamaño máximo del rostro
        )

        if len(faces) == 0:
            # No se detecta rostro
            return jsonify({'bpm': 'No se detecta el rostro'})

        # Procesamos la región de interés (frente) en el rostro detectado
        for (x, y, w, h) in faces:
            roi = gray[y:y+h//3, x:x+w]  # Solo la parte superior del rostro para la señal

        # Convertimos la región de interés en un vector para calcular el BPM
        signal = roi.mean(axis=0).flatten()
        
        # Aplicamos el filtro de paso de banda a la señal
        fs = 30.0  # Ajustar según la frecuencia de muestreo real
        lowcut = 0.75
        highcut = 3.0
        filtered_signal = bandpass_filter(signal, lowcut, highcut, fs, order=5)

        # Calculamos la FFT de la señal filtrada
        N = len(filtered_signal)
        freqs = np.fft.fftfreq(N, d=1/fs)
        fft_signal = fft(filtered_signal)
        fft_amplitude = np.abs(fft_signal)

        # Obtenemos el BPM como la frecuencia dominante en el espectro
        idx = np.argmax(fft_amplitude)
        bpm = freqs[idx] * 60.0

        # Limitamos el BPM a valores fisiológicamente realistas
        if bpm < 40 or bpm > 180:
            return jsonify({'bpm': 'Rango no válido'})

        return jsonify({'bpm': bpm})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Error en el procesamiento de la imagen'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
