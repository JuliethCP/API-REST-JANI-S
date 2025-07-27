# API REST - JANI-S

Este repositorio contiene la **API REST desarrollada en Python con Flask** para el proyecto **JANI-S (Jarvis Artificial con Reconocimiento Facial y de Voz)**. Esta API permite la conexión entre la interfaz desarrollada en React y los modelos de inteligencia artificial que impulsan las funcionalidades del asistente virtual.

## 🧠 Descripción

El propósito de esta API es proporcionar endpoints para interactuar con diversos modelos de IA entrenados para tareas como:

- Predicciones económicas (Bitcoin, S&P500, precio del aguacate)
- Análisis del crimen y predicción de accidentes cerebrovasculares
- Predicciones inmobiliarias y de COVID-19
- Clasificación de imágenes para reconocimiento emocional y detección de armas
- Reconocimiento facial y procesamiento de voz (en combinación con el frontend)

Los modelos fueron previamente entrenados y se cargan desde archivos `.pkl` o `.h5`.

## 🔧 Tecnologías utilizadas

- **Python 3**
- **Flask** y **Flask-CORS**
- **Joblib** y **Pickle** para manejo de modelos
- **TensorFlow** para modelos de clasificación de imágenes
- **Pandas**, **NumPy** y **Statsmodels** para procesamiento de datos
- **OpenCV** y **PIL** para manejo de imágenes

## 📌 Endpoints principales

- `POST /predict`: Realiza predicciones usando modelos según la variable proporcionada (ej: bitcoin, house, crimes, automobile, stroke, etc.)
- `POST /emotion`: Clasifica emociones humanas a partir de imágenes faciales
- `POST /weapons`: Detecta armas u objetos peligrosos en imágenes
