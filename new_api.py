from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# Cargamos el clasificador Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Obtenemos la imagen en Base64
        data = request.json['image']
        img_bytes = base64.b64decode(data)
        
        # Convertimos a una imagen compatible con OpenCV
        image = Image.open(io.BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convertimos la imagen a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectamos el rostro ajustando los parámetros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            # Si no se detecta rostro, devolvemos un mensaje de error
            return jsonify({'bpm': 'No se detecta el rostro'})

        # Si se detecta el rostro, tomamos la región de interés (por ejemplo, la frente)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h//3, x:x+w]  # Región superior del rostro para BPM

        # Convertimos la región de interés en un vector para calcular el BPM
        signal = roi.mean(axis=0).flatten()
        
        # Filtramos la señal con un filtro de paso de banda
        fs = 30.0  # Supuesto de frecuencia de muestreo
        lowcut = 0.75
        highcut = 3.0
        filtered_signal = bandpass_filter(signal, lowcut, highcut, fs, order=5)

        # Calculamos la FFT de la señal filtrada
        N = len(filtered_signal)
        freqs = np.fft.fftfreq(N, d=1/fs)
        fft_signal = fft(filtered_signal)
        fft_amplitude = np.abs(fft_signal)

        # Obtenemos el BPM como la frecuencia dominante
        idx = np.argmax(fft_amplitude)
        bpm = freqs[idx] * 60.0

        # Limitamos el rango de BPM a valores realistas
        if bpm < 40 or bpm > 180:
            return jsonify({'bpm': 'Rango no válido'})

        return jsonify({'bpm': bpm})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Error en el procesamiento de la imagen'}), 500

if __name__ == '__main__':
    app.run(debug=True)
